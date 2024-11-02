import pdb
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt

class PanaromaStitcher():
    def __init__(self):
        pass
    
    def get_resized_images(self,image_paths):
        images = [cv2.imread(im) for im in image_paths]
        resized_images = []
        target_height = 720  # Target height for resizing

        for idx, img in enumerate(images):
            # Get the original dimensions
            height, width = img.shape[:2]
            # if(height < 720):
            #     resized_images.append(img)
            #     continue
            print(f"Image {idx + 1}: Original Dimensions = {width}x{height}")
            
            # Calculate aspect ratio and new width based on target height
            aspect_ratio = width / height
            new_width = int(target_height * aspect_ratio)
            new_height = target_height
            new_dimensions = (new_width, new_height)
            
            # Resize the image while maintaining the aspect ratio
            resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
            resized_images.append(resized_img)
            print(f"Image {idx + 1}: Resized Dimensions = {new_width}x{new_height}")
        return resized_images

    def ProjectOntoCylinder(self,InitialImage, f=1100):

        if len(InitialImage.shape) == 2:
            InitialImage = cv2.cvtColor(InitialImage, cv2.COLOR_GRAY2BGR)
        
        h, w = InitialImage.shape[:2]
        center = np.array([w // 2, h // 2], dtype=np.float32)
        
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        theta = (x - center[0]) / f
        h_cyl = (y - center[1]) / f
        
        x_cyl = f * np.tan(theta)
        y_cyl = h_cyl * np.sqrt(x_cyl**2 + f**2)
        
        map_x = x_cyl + center[0]
        map_y = y_cyl + center[1]
        
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        TransformedImage = cv2.remap(InitialImage, 
                                    map_x, 
                                    map_y,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0)
        
        gray = cv2.cvtColor(TransformedImage, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        coords = cv2.findNonZero(mask)
        x_coords = coords[:, 0, 0]
        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        
        TransformedImage = TransformedImage[:, min_x:max_x+1]
        
        return TransformedImage

    def detectFeatures(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        return kp, des
    
    def matchFeatures(self,image1, image2):
        kp1, des1 = self.detectFeatures(image1)
        kp2, des2 = self.detectFeatures(image2)
    
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m in matches:
            if len(m) ==2 and m[0].distance < 0.75 * m[1].distance:
                good_matches.append((m[0].trainIdx, m[0].queryIdx, m[0].distance))
        good_matches = sorted(good_matches, key = lambda x:x[2])[:100]
        return kp1, kp2, good_matches
    
    def calculateHomography(self,src,dst):
        A = []
        for i in range(len(src)):
            x, y = src[i].flatten()
            u, v = dst[i].flatten()
            A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        H = H / H[2, 2]

        return H
    
    def homographRANSAC(self,kp1, kp2, matches, n_iter, tol):
        best_H = None
        best_mask = None
        best_score = 0
        for i in range(n_iter):
            randomIdx = np.random.choice(range(len(matches)), 4)
            random_matches = [matches[i] for i in randomIdx]
            
            src_pts = np.float32([kp1[i].pt for (_, i, _) in random_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[i].pt for (i, _, _) in random_matches]).reshape(-1, 1, 2)
            
            H = self.calculateHomography(src_pts, dst_pts)

            current_inliers = []
            for j in range(len(matches)):
                src = kp1[matches[j][1]].pt
                dst = kp2[matches[j][0]].pt
                src = np.array([src[0], src[1], 1]).reshape(3, 1)
                dst = np.array([dst[0], dst[1], 1]).reshape(3, 1)
                projected = np.dot(H, src)
                projected = projected / projected[2]
                error = np.linalg.norm(projected - dst)
                if error < tol:
                    current_inliers.append(j)
            
            if len(current_inliers) > best_score:
                best_score = len(current_inliers)
                best_mask = current_inliers
                best_H = H
        return best_H, best_mask
    
    def warpImage(self,image, H):
        length, width = image.shape[:2]
        image_corners = np.array(
            [[0, width - 1, 0, width - 1], [0, 0, length-1, length-1], [1, 1, 1, 1]]
        )
        warped_corners = np.dot(H, image_corners)
        warped_corners /= warped_corners[2,:]
        minx = np.floor(np.min(warped_corners[0]))
        miny = np.floor(np.min(warped_corners[1]))
        maxx = np.floor(np.max(warped_corners[0]))
        maxy = np.floor(np.max(warped_corners[1]))
        sizeX = int(maxx - minx)
        sizeY = int(maxy - miny)
        homoInv = np.linalg.inv(H)
        x_indices, y_indices = np.meshgrid(np.arange(sizeX) + minx, np.arange(sizeY) + miny)
        coordinates = np.stack((x_indices.flatten(), y_indices.flatten(), np.ones_like(x_indices.flatten())), axis=0)
        transformed_coords = homoInv @ coordinates
        transformed_coords /= transformed_coords[2, :]
        realx = transformed_coords[0, :].reshape(sizeY, sizeX)
        realy = transformed_coords[1, :].reshape(sizeY, sizeX)
        valid_mask = (realx >= 0) & (realx < width) & (realy >= 0) & (realy < length)
        warpedImg = np.zeros((sizeY, sizeX, 3), dtype=image.dtype)
        warpedImg[valid_mask] = image[realy[valid_mask].astype(int), realx[valid_mask].astype(int)]
        return warpedImg, minx, miny 
    
    def shrink_image_array(self,result):
        row_sums = np.sum(result, axis=1)
        col_sums = np.sum(result, axis=0)

        non_zero_rows = np.where(row_sums != 0)[0]
        non_zero_cols = np.where(col_sums != 0)[0]

        trimmed_result = result[non_zero_rows[0]:non_zero_rows[-1] + 1, non_zero_cols[0]:non_zero_cols[-1] + 1]

        return trimmed_result
    
    def blendImages(self,imageA, warpedimage, minx, miny, H):
        (hA, wA) = imageA.shape[:2]
        alpha=0.4
        (hB, wB) = warpedimage.shape[:2]
        minx = int(minx)
        miny = int(miny)
        x_max = int(max([wA, wB + abs(minx),wA + abs(minx)]))
        y_max = int(max([hA, hB + abs(miny),hA + abs(miny)]))
        result = np.zeros((y_max, x_max, 3), dtype=imageA.dtype)
        mask1 = np.zeros((hA, wA), dtype=np.uint8)
        mask2 = np.zeros((hB, wB), dtype=np.uint8)
        resulttemp = np.zeros((y_max, x_max, 3), dtype=imageA.dtype)
        if minx > 0:
            if (miny<0):
                result[:hB, minx : minx + wB] = warpedimage
                # print("This is miny", miny)
                mask = np.sum(result[-miny : -miny + hA, :wA], axis=2) == 0
                mask1 = 1-mask
                mask1 = np.where(mask1==0,1,0.5)
                # result[-miny : -miny + hA, :wA][mask] = imageA[mask]
                resulttemp[-miny : -miny + hA, :wA]=imageA
                mask2 = np.sum(resulttemp[:hB, minx : minx + wB], axis=2) == 0
                mask2 = 1-mask2
                mask2 = np.where(mask2==0,1,0.5)
                result[:hB, minx : minx + wB] = warpedimage * mask2[:,:,np.newaxis]
                result[-miny : -miny + hA, :wA] += (imageA * mask1[:,:,np.newaxis]).astype(np.uint8)
                # print("1")
                # # result[-miny : -miny + hA, :wA][inv_mask] = (alpha*imageA[inv_mask] + (1-alpha)*result[-miny : -miny + hA, :wA][inv_mask])
                # result[-miny : -miny + hA, :wA] += imageA//2
            else:
                result[miny : miny + hB, minx : minx + wB] = warpedimage
                mask = np.sum(result[:hA, :wA], axis=2) == 0
                mask1 = 1-mask
                mask1 = np.where(mask1==0,1,0.5)
                # result[:hA, :wA][mask] = imageA[mask]
                resulttemp[:hA, :wA]=imageA
                mask2 = np.sum(resulttemp[miny : miny + hB, minx : minx + wB], axis=2) == 0
                mask2 = 1-mask2
                mask2 = np.where(mask2==0,1,0.5)
                result[miny : miny + hB, minx : minx + wB] = warpedimage * mask2[:,:,np.newaxis]
                result[:hA, :wA] += (imageA * mask1[:,:,np.newaxis]).astype(np.uint8)
                # print("2")
                # result[:hA, :wA][inv_mask] = (alpha*imageA[inv_mask] + (1-alpha)*result[:hA, :wA][inv_mask])
                # result[:hA, :wA] += imageA//2
        else:
            # print("This is hA", hA,-miny)
            if(miny < 0):
                result[:hB,:wB] = warpedimage[:,:]
                # print("This is miny", miny)
                mask = np.sum(result[-miny : -miny + hA, -minx : wA - minx], axis=2) == 0
                mask1 = 1-mask
                mask1 = np.where(mask1==0,1,0.5)
    # Place imageA values in result only where the mask is True
                # result[-miny : -miny + hA, -minx : wA - minx][mask] = imageA[mask]
                resulttemp[-miny : -miny + hA, -minx : wA - minx]=imageA
                mask2 = np.sum(resulttemp[:hB, :wB], axis=2) == 0
                mask2 = 1-mask2
                mask2 = np.where(mask2==0,1,0.5)
                result[:hB, :wB] = warpedimage * mask2[:,:,np.newaxis]
                result[-miny : -miny + hA, -minx : wA - minx] += (imageA * mask1[:,:,np.newaxis]).astype(np.uint8)
                # print("3")
                # result[-miny : -miny + hA, -minx : wA - minx][inv_mask] = (alpha*imageA[inv_mask] + (1-alpha)*result[-miny : -miny + hA, -minx : wA - minx][inv_mask])
                # result[-miny:-miny+hA,-minx:wA-minx] += imageA[:,:]//2
            else:
                result[miny:hB+miny, :wB] = warpedimage[:,:]
                mask = np.sum(result[:hA, -minx : wA - minx], axis=2) == 0
                mask1 = 1-mask
                mask1 = np.where(mask1==0,1,0.5)
    # Apply imageA values to result only where the mask is True
                # result[:hA, -minx : wA - minx][mask] = imageA[mask]
                resulttemp[:hA, -minx : wA - minx]=imageA
                mask2 = np.sum(resulttemp[miny : miny + hB, :wB], axis=2) == 0
                mask2 = 1-mask2
                mask2 = np.where(mask2==0,1,0.5)
                result[miny : miny + hB, :wB] = warpedimage * mask2[:,:,np.newaxis]
                result[:hA, -minx : wA - minx] += (imageA * mask1[:,:,np.newaxis]).astype(np.uint8)
                # print("4")
                # result[:hA, -minx : wA - minx][inv_mask] = (alpha*imageA[inv_mask] + (1-alpha)*result[:hA, -minx : wA - minx][inv_mask])
                # result[:hA, -minx : wA - minx] += imageA[:,:]//2
        # plt.imshow(mask1, cmap='gray', interpolation='nearest')
        # plt.colorbar()  # Optional: adds a color bar to indicate the scale
        # plt.show()
        # plt.imshow(mask2, cmap='gray', interpolation='nearest')
        # plt.colorbar()  # Optional: adds a color bar to indicate the scale
        # plt.show()
        result = self.shrink_image_array(result)
        return result
    
    def stitchcustom(self,image0, image1):
        at, bt, ct = self.matchFeatures(image0, image1)
        ht,_ = self.homographRANSAC(at, bt, ct, 3000, 5.0)
        # htn = np.dot(prevhomo,ht)
        warpedd, ax, ay = self.warpImage(image0, ht)
        result = self.blendImages(image1, warpedd, ax, ay, ht)
        # result = trim(result)
        # plt.figure(figsize=(20, 20))
        # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.show()
        return result
    
    def make_panaroma_for_images_in(self,path,useCylin=True):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        print(all_images)
        resized_images = self.get_resized_images(all_images)
        warped_images = []
        focal_length = 1000
        imageset = all_images[0].split(os.sep)[-2]
        if(imageset == 'I1'):
            focal_length = 2100
        for image in resized_images:
            warped_images.append(self.ProjectOntoCylinder(image,focal_length=focal_length))
        # warped_images = self.cylindrical_warp(resized_images)
       
        if(imageset == 'I3'):
            resized_images.pop(0)
            warped_images.pop(0)
            resized_images.pop(0)
            warped_images.pop(0)
        if(useCylin):
            resized_images = warped_images
        mididx = len(resized_images)//2
        result = warped_images[mididx]
        left = mididx - 1
        right = mididx + 1
        while(left >= 0 or right < len(resized_images)):
            if left >= 0:
                result = self.stitchcustom(resized_images[left], result)
                left -= 1
            if right < len(resized_images):
                result = self.stitchcustom(resized_images[right], result)
                right += 1
        # Collect all homographies calculated for pair of images and return
        # homography_matrix_list =[]
        # # Return Final panaroma
        # stitcher = cv2.Stitcher_create()
        # status, stitched_image = stitcher.stitch([cv2.imread(im) for im in all_images])
        # stitched_image = cv2.imread(all_images[0])
        #####
        
        return result, []