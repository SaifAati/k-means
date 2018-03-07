import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

def load_image(image_title):
	image = img.imread(image_title)
	plt.imshow(image)
	#plt.show()
	return image


def initialization(image,k):
    # Choose k centroid randomly
    shape = np.shape(image)

    #choose random coordinates
    coord_l = np.random.choice(shape[0],k)
    coord_c = np.random.choice(shape[1],k)

    # Define a centoid matrix
    centroid_mat = np.zeros((k, shape[2]))
    for i in range(0,k):
        for j in range(0,shape[2]):
            centroid_mat[i,j] = image[coord_l[i],coord_c[i],j]
    print("centroid_mat=\n",centroid_mat)
    return centroid_mat

def assignment_classification(image,centroid_mat,k):
    shape = np.shape(image)
    #==========================================================================================================
    """
    We assign for each pixel the closest cluster = classify the differnt pixels of the image  to be processed
    #INPUT:
        - Image to be processed
        - Centroid matrix 
        - Number of clusters: k
        
    #OUTPUT:
        - Classified matrix
        - nb_pixel_cluster: the number of pixels for the same centroid
    """
    # ==========================================================================================================

    classified_mat = np.zeros((shape[0], shape[1]))
    for i in range (shape[0]):
        for j in range (shape[1]):
            dist_ = []
            for k_ in range (k):
                #Calculating the Euclidean Distance
                dist_.append(np.sqrt(((centroid_mat[k_,0]-image[i,j,0])**2)+((centroid_mat[k_,1]-image[i,j,1])**2)+
                                     ((centroid_mat[k_,2]-image[i,j,2])**2)))

            mn_, idx = min((dist_[m], m) for m in range(len(dist_)))
            classified_mat[i,j] = idx

    print("classified_mat=\n", classified_mat)

    # calculate the number of pixels for the same centroid
    nb_pixel_cluster = np.zeros((k, 1))
    for k_ in range(k):
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (classified_mat[i, j] == k_):
                    nb_pixel_cluster[k_, 0] += 1
    print("nb_pixel_for each_cluster=\n", nb_pixel_cluster)
    return classified_mat, nb_pixel_cluster


def re_estimate_matrix (image, classified_mat,k,nb_pixel_cluster):

    # ==========================================================================================================
    """
    Update the Centroid matrix and calculate the number of differences between the old and updated classifed matrix 
    #INPUT:
        - Image to be processed
        - classified_mat (old one)
        - Number of clusters: k

    #OUTPUT:
        - New centroid matrix
        - New Classified matrix
        - number of differences between the old and updated classifed matrix
        - The updated number of pixels for the same centroid
        
    """
    # ==========================================================================================================
    shape = np.shape(image)
    # Define a new centoid matrix
    centroid_mat_update = np.zeros((k, shape[2]))
    for k_ in range(k):
        for i in range (shape[0]):
            for j in range(shape[1]):
                if(classified_mat[i,j]==k_):
                    for color in range(shape[2]):
                        centroid_mat_update[k_,color] = centroid_mat_update[k_,color] + image[i,j,color]

    for i in range(k):
        for j in range(shape[2]):
            centroid_mat_update[i,j] =centroid_mat_update[i,j]/nb_pixel_cluster[i,0]

    print("centroid_mat_update=\n", centroid_mat_update)

    # Update the Classification matrix
    centroid_mat = centroid_mat_update
    classified_mat_update = np.zeros((shape[0], shape[1]))
    classified_mat_update,nb_pixel_cluster_update = assignment_classification(image=image, centroid_mat = centroid_mat_update, k=k)

    print("classified_mat_update=\n", classified_mat_update)

    # calculate the number of differences between the old and updated classifed matrix
    num_of_diff = 0
    for i in range (shape[0]):
        for j in range(shape[1]):
            if (classified_mat_update[i,j]!=classified_mat[i,j]):
                num_of_diff +=1
    print("num_of_diff=",num_of_diff)

    return centroid_mat_update,classified_mat_update,nb_pixel_cluster_update,num_of_diff

def k_means(image,max_iteration_number,thershold_diff,k):
    # ==========================================================================================================
    """
    Update the Centroid matrix and calculate the number of differences between the old and updated classifed matrix
    #INPUT:
        - Image to be processed             : image
        - The maximum number of iteration   : max_iteration_number
        - Threshold on the number of differences between the different classification matrix : thershold_diff
        - Number of clusters: k
    #OUTPUT:
        - New centroid matrix

    """
    # ==========================================================================================================
    #Step1 : initialization
    print("\n-----Initialization------\n")
    centroid_mat = initialization(image=image, k=k)

    #Step 2 : We assign for each pixel the closest cluster = classify the differnt pixels of the image  to be processed
    print("\n----- Step2: assigning for each pixel the closest cluster ------\n")
    classified_mat, nb_pixel_cluster = assignment_classification(image=image, centroid_mat=centroid_mat, k=k)

    #Step 3 : iteration
    print("\n----- Step3: Iteration ------\n")
    num_iteration = 0
    while (num_iteration < max_iteration_number):
        print("\n-----Iteration number %d----------\n"%num_iteration)
        # Update the Centroid matrix and calculate the number of differences between the old and updated classifed matrix
        centroid_mat_update,classified_mat_update, nb_pixel_cluster_update, num_of_diff = \
            re_estimate_matrix(image=image, classified_mat=classified_mat, k=k, nb_pixel_cluster=nb_pixel_cluster)
        if (num_of_diff > thershold_diff):
            num_iteration+=1
            classified_mat = classified_mat_update
            nb_pixel_cluster = nb_pixel_cluster_update
        if(num_of_diff < thershold_diff):
            print("The difference threshold is reached")
            break

    print(classified_mat_update)
    return classified_mat_update*200/5

    return

if __name__ == '__main__':
    image1 = "crop.tif"
    image2 = "dalleIRC.tif"
    image3 = "dalleRVB.tif"

    # Load the image
    image = load_image(image_title= image3)
    shape = np.shape(image)
    print("shape of the image",shape)
    # Number of K clusters
    k= 4
    # Stopping criteria
    max_iteration_number = 20

    thershold_diff = 1000

    #The k-means algorithm
    final_classif_mat = k_means(image=image, max_iteration_number= max_iteration_number, thershold_diff =thershold_diff , k=k)

    print("final_classif_mat=\n",final_classif_mat)
    plt.imshow(final_classif_mat)
    plt.colorbar()
    plt.show()
    title = image3+"_classified.png"
    plt.imsave(title,final_classif_mat)




