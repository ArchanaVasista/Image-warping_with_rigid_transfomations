
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

#Implementation of class RigidTransform, which specifies a 3D rigid transformation which can warp 3D image volumes.
class RigidTransform:
# Class constructor __init__ function, which takes a set of 6 rigid transformation parameters, 3 rotations and 3 translations, as input. 
    def __init__(self, rotation, translation, warp_im_size, flag=False):

# flag by default is set to False
        self.flag = flag
# Rotation along x, y and z axes
        self.rot_x = rotation[0]
        self.rot_y = rotation[1]
        self.rot_z = rotation[2]
# Translation along x, y and z axes
        self.tr_x = translation[0]
        self.tr_y = translation[1]
        self.tr_z = translation[2]
# Warped image size
        self.warp_im_size_x = warp_im_size[0]
        self.warp_im_size_y = warp_im_size[1]
        self.warp_im_size_z = warp_im_size[2]
# Precompute a rotation matrix and a translation vector and stored in the returned class object.        
# Rotation matrices along x, y and z axes       
        theta_x = np.array([[1,0,0],[0, np.cos(self.rot_x),-np.sin(self.rot_x)],[0,np.sin(self.rot_x),np.cos(self.rot_x)]])
        theta_y = np.array([[np.cos(self.rot_y), 0, np.sin(self.rot_y)],[0,1,0],[-np.sin(self.rot_y),0,np.cos(self.rot_y)]])
        theta_z = np.array([[np.cos(self.rot_z),-np.sin(self.rot_z),0],[np.sin(self.rot_z),np.cos(self.rot_z),0],[0,0,1]])

#R is rotation vector
#T is translation vector
        self.R = np.matmul(theta_z, np.matmul(theta_y, theta_x))
        self.T = np.array([self.tr_x, self.tr_y, self.tr_z])

# self.ddf to calculate new volume when the flag condition is true 
# ddf is precomputed in __init__ and stored in the returned class object
        self.compute_ddf((self.warp_im_size_x, self.warp_im_size_y, self.warp_im_size_z)) 
# When the flag is set to True the compose function combines ddfs instead of composing tranlation and rotation vectors
        if self. flag == True:
            self.compose = self.composing_ddfs

# Implement a class member function compute_ddf, which takes a three-item tuple, representing the warped image size
# dense displacement field vector            
    def compute_ddf(self, warp_im_size):
# three item tuple warped image size
        self.warp_im_size_x, self.warp_im_size_y, self.warp_im_size_z = warp_im_size

# Take inverse of rotation matrix
        x = np.linalg.inv(self.R)
# preallocation of dense displacement vector in an empty list
        self.ddf = []
# compute the ddf by multiplying the array by inverse rotation matrix and subtract it with translation vector
        for i in range (image_volume.shape[0]):
            for j in range (image_volume.shape[1]):
                for k in range(image_volume.shape[2]):
#Update ddf
                    self.ddf.append(np.matmul(x, np.array([i, j, k]))-self.T)
        self.ddf = np.array(self.ddf)
# ddf returns a NumPy array that defines a 3D displacement vector
        return self.ddf        
# Implementation of a class member function warp, which takes a NumPy array, representing a 3D image volume, as input.
    def warp (self, image_volume):
# preallocation of warped volume            
# if the flag is set to true while calling the function the warping is carried out with composed ddfs 
       if self. flag == True:
            warp_coords = self.ddf
# The flag is set to false by default, so when the function is called the warping is carried out with rotational and translational vectors     
       else:
# Pre allocate an empty list for warped coordinates
            warp_coords = []
            for n in range (image_volume.shape[0]):
                for m in range (image_volume.shape[1]):
                    for l in range(image_volume.shape[2]):
# Update warped coordinates by applying rotation and tarnslation
                        warp_coords.append(np.matmul(self.R, np.array([n,m,l])) + self.T)
# Change the warped coordinates to numpy array
       warp_coords = np.array(warp_coords)

# map_coordinate Maps the input array to specifed coordinates by interpolation.
# The common interpolation method is Nearest Neighbour
       warp_vol = map_coordinates(image_volume, [warp_coords.T[0], warp_coords.T[1], warp_coords.T[2]], order=1, mode= 'nearest', cval=np.NaN, prefilter=False)
       warp_vol = warp_vol.reshape(image_volume.shape)
# this function returns warped image array
       return warp_vol

#Implementation of a class member function compose, which takes a second set of rigid transformation parameters as input       
    def compose(self, N_rotation, N_translation):
        
# compose returns a RigidTransform object 
        comp_trans = RigidTransform(N_rotation, N_translation, warp_im_size=(128, 128, 32)) 
# The transformations can be composed by matrix multiplication of rotational and translational vectors
        comp_trans_R = np.matmul(N_rotation, self.R)
        comp_trans_T = np.matmul(N_translation, self.T) + np.array(N_translation)
# updates the ddf after composing        
        comp_trans.compute_ddf((self.warp_im_size_x, self.warp_im_size_y, self.warp_im_size_z))            
# This function returns a rigid transformation object that combines the two rigid transformations         
        return comp_trans, comp_trans_R, comp_trans_T

# if the flag is set to true while calling the function the dense displacement field is computed by composing
# two ddfs and the warping is carried out with composed ddfs    
    def composing_ddfs(self, DDF1, DDF2):
# This function returns composed ddfs        
        return (DDF1, DDF2)
                            
#Load the image volume
image_volume = np.load ('image_train00.npy').T
# Values of rotational and translational vectors
R1 = np.array([1,2,3])*np.pi/180
T1 = np.array([0.2,0.4,0.6])
R2 = np.array([2,4,6])*np.pi/180
T2 = np.array([0.2,0.4,0.6])
R3 = np.array([5,7,9])*np.pi/180
T3 = np.array([0.2,0.4,0.6])
###########################################################################################
# Experiment 1
# warping under Transformation_T1
rigid_transform_T1 = RigidTransform(rotation=R1, translation=T1, warp_im_size=(128,128,32))
image_T1 = rigid_transform_T1.warp(image_volume)
# Compute DDF
DDF_T1 = rigid_transform_T1.compute_ddf(warp_im_size=(128,128,32))
# Specify the slices to be saved
x = [25,26,27,28,29] 
y = [0,1,2,3,4]  
# saving 5 slices
for i, j in zip(x, y):  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image_volume[:,:,i].T, cmap='turbo')
    ax1.set_title(f'Axial slice{j + 1}'' image_volume')
    ax2.imshow(image_T1[:,:,i].T, cmap='turbo')
    ax2.set_title(f'Axial slice{j + 1}'' image_T1')
    fig.suptitle('Task_3 Experiment 1')
    fig.savefig(f'Task3_Exp1_T1_Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')

   
# warping under Transformation_T2
rigid_transform_T2 = RigidTransform(rotation=R2, translation=T2, warp_im_size=(128,128,32))
# Compute DDF
image_T2 = rigid_transform_T2.warp(image_volume)
DDF_T2 = rigid_transform_T2.compute_ddf(warp_im_size=(128,128,32))

# warping under Transformation_T3
rigid_transform_T3 = RigidTransform(rotation=R3, translation=T3, warp_im_size=(128,128,32))
image_T3 = rigid_transform_T3.warp(image_volume)
# compute DDF
DDF_T3 = rigid_transform_T3.compute_ddf(warp_im_size=(128,128,32))

#composed transformation T1xT2
# call compose function to obtain composed transform T1xT2
compose_T1xT2 =rigid_transform_T1.compose(N_rotation=R2, N_translation=T2)
# extract composed rotational vector
R1xR2 = compose_T1xT2[1]
# extract composed translational vector
T1xT2 = compose_T1xT2[2]
# using T1xT2 to obtain composed image
rigid_transform_T1xT2 = RigidTransform(rotation=R1xR2, translation=T1xT2, warp_im_size=(128,128,32))
# compute composed DDF of T1 and T2 transformation
DDF_T1xT2 = rigid_transform_T1xT2.compute_ddf(warp_im_size=(128,128,32))
comp_image_T1xT2 = rigid_transform_T1xT2.warp(image_volume)

# Save the slices
for i, j in zip(x, y):  
    fig, (ax) = plt.subplots(figsize=(10, 5))
    ax.imshow(comp_image_T1xT2[:,:,i].T, cmap='turbo')
    ax.set_title(f'Axial slice{j + 1}')
    fig.suptitle('Task_3 Experiment 1-Composed transform T1xT2')
    fig.savefig(f'Task3_Exp1_T1xT2_comp Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')

# sequential transformation on previously warped image by T1 transformation.
# Transformations applied from right to left T2(T1(image))
image_seq_T1_T2 = rigid_transform_T2.warp(image_T1)
# save the slices
for i, j in zip(x, y):  
    fig, (ax) = plt.subplots(figsize=(10, 5))
    ax.imshow(image_seq_T1_T2[:,:,i].T, cmap='turbo')
    ax.set_title(f'Axial slice{j + 1}')
    fig.suptitle('Task_3 Experiment 1-Sequential transform T1_T2')
    fig.savefig(f'Task3_Exp1_T1_T2_seq Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')

# comparison between composed and sequential transforms T1xT2 and T1_T2
# The two tranformations have similar effects
fig, (ax1, ax2) = plt.subplots(1,2,figsize =(10,5))
ax1.imshow(comp_image_T1xT2[:,:,29].T, cmap='turbo')
ax1.set_title('Composed_image_T1xT2')
ax2.imshow(image_seq_T1_T2[:,:,29].T, cmap='turbo')
ax2.set_title('Sequential_image_T1_T2')
fig.suptitle('Task_3 Experiment 1-comparison between composed and sequential transforms')
fig.savefig('Task3_comparison between T1xT2 and T1_T2.png',dpi=300,bbox_inches='tight')

#call compose function to obtain composed transform T1xT2xT3
composeT1xT2xT3 = rigid_transform_T1xT2.compose(N_rotation=R3, N_translation=T3)
# extract composed rotational vector
R1xR2xR3 = composeT1xT2xT3[1]
# extract composed translational vector
T1xT2xT3 = composeT1xT2xT3[2]
# using T1xT2xT3 to obtain composed image
rigid_transform_T1xT2xT3 = RigidTransform(rotation=R1xR2xR3, translation=T1xT2xT3, warp_im_size=(128,128,32))
comp_image_T1xT2xT3 = rigid_transform_T1xT2xT3.warp(image_volume)

# Save the slices
for i, j in zip(x, y):  
    fig, (ax) = plt.subplots(figsize=(10, 5))
    ax.imshow(comp_image_T1xT2xT3[:,:,i].T, cmap='turbo')
    ax.set_title(f'Axial slice{j + 1}')
    fig.suptitle('Task_3 Experiment 1-Composed transform T1xT2xT3')
    fig.savefig(f'Task3_Exp1_T1xT2xT3_comp Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')

# sequential transformation on previously warped image by T1_T2 transformation.
image_seq_T1_T2_T3 = rigid_transform_T3.warp(image_seq_T1_T2)

#Save the slices
for i, j in zip(x, y):  
    fig, (ax) = plt.subplots(figsize=(10, 5))
    ax.imshow(image_seq_T1_T2_T3[:,:,i].T, cmap='turbo')
    ax.set_title(f'Axial slice{j + 1}')
    fig.suptitle('Task_3 Experiment 1-Sequential transform T1_T2_T3')
    fig.savefig(f'Task3_Exp1_T1_T2_T3_seq Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')

# comparison between composed and sequential transforms T1xT2xT3 and T1_T2_T3
# The two tranformations have similar effects
fig, (ax1, ax2) = plt.subplots(1,2,figsize =(10,5))
ax1.imshow(comp_image_T1xT2xT3[:,:,29].T, cmap='turbo')
ax1.set_title('Composed_image_T1xT2xT3')
ax2.imshow(image_seq_T1_T2_T3[:,:,29].T, cmap='turbo')
ax2.set_title('Seq_image_T1_T2_T3')
fig.suptitle('Task_3 Experiment 1-comparison between composed and sequential transforms')
fig.savefig('Task3_comparison between T1xT2xT3 and T1_T2_T3',dpi=300,bbox_inches='tight')

####################################################################################################
# Experiment 2 composing ddfs 
#composing_ddfs T1xT2
# Use the composed rotational and translational vectors to obtain new rigid transform
sec_rigid_transform_T1xT2 = RigidTransform(rotation=R1xR2, translation=T1xT2, warp_im_size=(128,128,32),flag=True)
# Compose the two DDFs DDF_T1 and DDF_T2 by composing_ddfs function
sec_DDF_T1xT2 = sec_rigid_transform_T1xT2.compose(DDF1=DDF_T1, DDF2=DDF_T2)
# obtain the warped image
new_comp_image_T1xT2 = sec_rigid_transform_T1xT2.warp(image_volume)

#Save the slices
for i, j in zip(x, y):  
    fig, (ax) = plt.subplots(figsize=(10, 5))
    ax.imshow(new_comp_image_T1xT2[:,:,i].T, cmap='turbo')
    ax.set_title(f'Axial slice{j + 1}')
    fig.suptitle('Task_3 Experiment 2-Composed transform T1xT2 by composing DDFs')
    fig.savefig(f'Task3_Exp2_T1xT2_comp_ddf Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')

#composing_ddfs T1xT2xT3
# Use the composed rotational and translational vectors to obtain new rigid transform
sec_rigid_transform_T1xT2xT3 = RigidTransform(rotation=R1xR2xR3, translation=T1xT2xT3, warp_im_size=(128,128,32), flag=True)
# Compose the two DDFs DDF_T1xT2 and DDF_T3 by composing_ddfs function
DDF_T1xT2xT3 = sec_rigid_transform_T1xT2.compose(DDF1=DDF_T1xT2, DDF2=DDF_T3)
# obtain the warped image
new_comp_image_T1xT2xT3 = sec_rigid_transform_T1xT2xT3.warp(image_volume)

#Save slices
for i, j in zip(x, y):  
    fig, (ax) = plt.subplots(figsize=(10, 5))
    ax.imshow(new_comp_image_T1xT2xT3[:,:,i].T, cmap='turbo')
    ax.set_title(f'Axial slice{j + 1}')
    fig.suptitle('Task_3 Experiment 2-Composed transform T1xT2xT3 by composing DDFs')
    fig.savefig(f'Task3_Exp2_T1xT2xT3_comp_ddf Axial slice_{j+1}.png',dpi=300,bbox_inches='tight')

#comparison between composed transforms for T1 and T2 with and without composed_ddfs
fig, (ax1, ax2) = plt.subplots(1,2,figsize =(10,5))
ax1.imshow(comp_image_T1xT2[:,:,29].T, cmap='turbo')
ax1.set_title('Without_composing_ddfs_T1xT2')
ax2.imshow(new_comp_image_T1xT2[:,:,29].T, cmap='turbo')
ax2.set_title('With_composing_ddfs_T1xT2')
fig.suptitle('Task_3 Experiment 2-comparison between composed transforms with and without composed_ddfs')
fig.savefig('Task3_Exp2_comparison with and without composed_ddfs_T1xT2.png',dpi=300,bbox_inches='tight')

#comparison between composed transforms for T1, T2 and T3 with and without composed_ddfs
fig, (ax1, ax2) = plt.subplots(1,2,figsize =(10,5))
ax1.imshow(comp_image_T1xT2xT3[:,:,29].T, cmap='turbo')
ax1.set_title('Without_composing_ddfs_T1xT2xT3')
ax2.imshow(new_comp_image_T1xT2xT3[:,:,29].T, cmap='turbo')
ax2.set_title('With_composing_ddfs_T1xT2xT3')
fig.suptitle('Task_3 Experiment 2-comparison between composed transforms with and without composed_ddfs')
fig.savefig('Task3_Exp2_comparison with and without composed_ddfs_T1xT2xT3.png',dpi=300,bbox_inches='tight')

#####################################################################################################
mean_difference1 = np.mean(new_comp_image_T1xT2-comp_image_T1xT2)
print(f'Mean difference between two composing algorithms with and without using composing_ddfs for T1 and T2:{mean_difference1}')

standard_deviation_difference1 = np.std(new_comp_image_T1xT2-comp_image_T1xT2)
print(f'Standard deviation difference between two composing algorithms with and without using composing_ddfs for T1 and T2:{standard_deviation_difference1}')

mean_difference2 = np.mean(new_comp_image_T1xT2xT3-comp_image_T1xT2xT3)
print(f'Mean difference between two composing algorithms with and without using composing_ddfs for T1, T2 and T3:{mean_difference2}')

standard_deviation_difference2 = np.std(new_comp_image_T1xT2xT3-comp_image_T1xT2xT3)
print(f'Standard deviation difference between two composing algorithms with and without using composing_ddfs for T1,T2 and T3:{standard_deviation_difference2}')
