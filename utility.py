
import SimpleITK as sitk
import pylab as plt
import matplotlib.animation as animation


# from medpy.io._tmp import dtype
# from Cython.Utility.MemoryView import shape

total_voxels = 2446272000 #just for knowledge 
total_patients = 274

BRATS_2015_parent_file = "/media/Data/Ahmed/new_batch_halawany"


def read_mha_image(image_path):
    """
    load mha image
    """
#     print image_path
    return sitk.ReadImage(image_path)


def read_mha_image_as_nuarray(image_path):
    """
    read mha image and return 3d numpy array
    """
    return sitk.GetArrayFromImage(read_mha_image(image_path))
   
def save_nuarray_as_mha(image_path, numpy_array):
    sitk.WriteImage(sitk.GetImageFromArray(numpy_array), image_path, True)
    
    


def draw_MRI(image_3d_gray, interval = 50, repeat_delay=500):
    """
    draws a 3d image in time laps
    image_3d_gray : 3d numpy array of gray scale
    """
    fig = plt.figure()
    ims = []
    for i in range(image_3d_gray.shape[0]):
        im = plt.imshow(image_3d_gray[i], cmap = "Greys_r")
        ims.append([im])
        
    
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=False, repeat_delay=repeat_delay) 
    plt.show()
    
def show_image(img, strg = ""):
    plt.imshow(img, cmap = "Greys_r")
    plt.text(0, 0, strg, color = "r")
    plt.show()   
    
    
    
    
    
    
    
    
    
    
    
    
