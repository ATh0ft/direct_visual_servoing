import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

from tqdm import tqdm

from spatialmath.base import rotx, roty, rotz

pixel_size = 0.003 #mm

scale_factor = 1
intrinsics=[1280, 800, 0.72507,0.72507, 1280/2 * pixel_size, 800/2*pixel_size]

intrinsics_matrix = np.array([[ scale_factor * intrinsics[2], 0,                            intrinsics[4], 0],
                              [ 0,                            scale_factor * intrinsics[3], intrinsics[5], 0],
                              [ 0,                            0,                            1,             0]])









def project_image_matrix(image, intrinsics, external, initial_dist = 200):
    image = image.astype(np.float32)

    boarder = 400
    
    width = 1280
    height = 800
    
    
    camera_matrix = np.matmul(intrinsics, external)
    
    # print(camera_matrix)

    expanded_image = np.zeros((height, width, 3), np.float32)


    #find image centers
    input_center = (np.array(list(image.shape))/2).astype(np.uint)
    
    focal_length_mm = 0.72507

    # encode pixel coordinates onto the image 
    pixel_index = np.array(np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])))
    
    # print(pixel_index)

    mm_index = (pixel_index - np.array([input_center[1], input_center[0]]).reshape(2, 1, 1))*pixel_size # the location on the image sensor in mm 

    # print(mm_index)

    points_in_camera_frame = (mm_index * initial_dist)/focal_length_mm
    
    z_layer = np.ones((1,points_in_camera_frame.shape[1], points_in_camera_frame.shape[2])) * initial_dist

    points_in_camera_frame = np.vstack([points_in_camera_frame, z_layer])
    
    points_in_camera_frame = np.vstack([points_in_camera_frame, np.ones((1,points_in_camera_frame.shape[1], points_in_camera_frame.shape[2])) ])
    

    # print(f"intrisics matrix shape {intrinsics.shape}")

    # print(points_in_camera_frame[0,:,:])

    points_on_sensor_mm =  np.matmul(points_in_camera_frame.transpose((1,2,0)), camera_matrix.T)

    points_on_sensor_mm[:,:,0] = points_on_sensor_mm[:,:,0] / points_on_sensor_mm[:,:,2]
    points_on_sensor_mm[:,:,1] = points_on_sensor_mm[:,:,1] / points_on_sensor_mm[:,:,2]

    points_on_sensor_pixels = (points_on_sensor_mm / pixel_size).astype(np.int32)

    # print(points_on_sensor_mm[:,:,0])



    # print(points_on_sensor_pixels[:,:,0])

    # print(points_on_sensor_mm.shape)

    # Map intensity values from original image to projected image
    
    points_on_sensor_pixels[points_on_sensor_pixels[:,:,1] > expanded_image.shape[0]-1, 1] = 0 
    points_on_sensor_pixels[points_on_sensor_pixels[:,:,1] < 0, 1] = 0 
    
    points_on_sensor_pixels[points_on_sensor_pixels[:,:,0] > expanded_image.shape[1]-1, 0] = 0 
    points_on_sensor_pixels[points_on_sensor_pixels[:,:,0] < 0, 0] = 0 

    # print(np.max(points_on_sensor_pixels[:,:,1]))


    
    expanded_image[points_on_sensor_pixels[:,:,1], points_on_sensor_pixels[:,:,0],:] = image[pixel_index[1,:,:], pixel_index[0,:,:],:]

    # plt.imshow(expanded_image.astype(np.uint8))
    #
    # plt.show()
    
    return expanded_image 


def image_delta(image_a, image_b):
    return np.sum(np.abs(image_a - image_b))


def image_delta_image(image_a, image_b):
    delta_image = np.abs(image_a - image_b)
    delta_image = (254*(delta_image/np.max(delta_image))).astype(np.uint8)
    return delta_image

def transformation_optimizer(desired_image, current_image, turn_rate, step_rate):
    
    initial_orientation = [0,0,0]
 
    initial_translation = [0,0,0] # the change in translation 

    orientation = np.array(initial_orientation)
    
    translation = np.array(initial_translation)


    orientation_test_kernel = np.array([[turn_rate, 0, 0],[0, turn_rate, 0], [0, 0, turn_rate],[-turn_rate, 0, 0], [0, -turn_rate, 0], [0, 0, -turn_rate]])
    translation_test_kernel = np.array([[step_rate, 0, 0],[0, step_rate, 0], [0, 0, step_rate],[-step_rate, 0, 0], [0, -step_rate, 0], [0, 0, -step_rate]])


    for step in range(100):
        
        test_orientations = np.ones_like(orientation_test_kernel)*orientation + orientation_test_kernel
        
        test_translations = np.ones_like(translation_test_kernel)*translation + translation_test_kernel

        


        orientation_results = []
        
        for test_orientation in test_orientations:
            R = rotx(test_orientation[0]) @ roty(test_orientation[1]) @ rotz(test_orientation[2])

            external_camera_matrix = np.zeros((4,4))    
            external_camera_matrix[0:3,0:3] = R.T
            external_camera_matrix[0:3,3] = np.matmul(-R.T, np.array(translation))
            external_camera_matrix[3,3] = 1

            projeted_image = project_image_matrix(desired_image,intrinsics_matrix,external_camera_matrix)
            
            orientation_results.append(image_delta(projeted_image, current_image))
            
        orientation = test_orientations[orientation_results.index(min(orientation_results))]

        # print(orientation)

        translation_results = []
        
        for test_translation in test_translations:
            R = rotx(orientation[0]) @ roty(orientation[1]) @ rotz(orientation[2])

            external_camera_matrix = np.zeros((4,4))    
            external_camera_matrix[0:3,0:3] = R.T
            external_camera_matrix[0:3,3] = np.matmul(-R.T, np.array(test_translation))
            external_camera_matrix[3,3] = 1

            projeted_image = project_image_matrix(desired_image,intrinsics_matrix,external_camera_matrix)
            
            translation_results.append(image_delta(projeted_image, current_image))
        
        translation = test_translations[translation_results.index(min(translation_results))]

        print([translation*0.001, orientation*180/np.pi])
        
        plt.imshow(image_delta_image(projeted_image, current_image))
        plt.pause(0.001)
            
    return [translation, orientation]
    


         
        





if __name__ == "__main__":
    ds_img_path = "/home/a/seasony/Germination_Room_v2/germ_room_desired.png"
    test_img_path = "/home/a/seasony/dataset_rotations/dataset/rgb/rgb_image_100.png"

    ds_img = cv2.imread(ds_img_path)

    test_img = cv2.imread(test_img_path)

    transformation_optimizer(ds_img, test_img, 0.004, 1)

  #   camera_trasform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 300], [0, 0, 0, 1]])
  #   
  # 
  #
  #   rot = rotz(0)
  #
  #
  #   external_camera_matrix = np.zeros((4,4))
  #
  #   external_camera_matrix[0:3,0:3] = rot.T
  #   external_camera_matrix[0:3,3] = np.matmul(-rot.T, np.array([0,0, -100]))
  #   external_camera_matrix[3,3] = 1
  #   
  #   print(external_camera_matrix.shape)
  #
  #   projected_image = project_image_matrix(ds_img, intrinsics_matrix, external_camera_matrix)
  #
  #
  #   print(intrinsics_matrix)
  #   
  #   plt.imshow(projected_image.astype(np.uint8))
  #
  #   plt.show()


    # find_transformation_with_dvs(test_img, ds_img, 0.01)
