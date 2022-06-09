def find_trial_peaks(all_vs, plot_path, subject_id, plot_size=(15,4), show_plots=True):
    '''
    Author: Ali Banijamali (banijamali.s@northeastern.edu)
    
    Input:
    all_vs: Must be a list containing 4 dataframes in this order:
    0: Left Macular Data
    1: Left Peripheral Data
    2: Right Macular Data
    3: Right Peripheral Data
    plot_size: (width, height)
    (!) It is possible to define more criteria for peak detection
    but currently it just finds all the peaks w/o any criteria.
    plot_path: a folder (DAVEP Analysis), will be created here for
    all results and plots.
    show_plots: TRUE/FALSE, whether to show the plots after running or not.
    
    Function:
    finds peaks of responses and saves the plots of their locations.
    
    Output:
    The function saves the peak location plots in "DAVEP Analysis" folder
    and it returns a list containing 4 dataframes in this order:
    0: Left Macular peak info
    1: Left Peripheral peak info
    2: Right Macular peak info
    3: Right Peripheral peak info
    Each one of these dataframes has the location(min, ms) and amplitude
    of the peaks of all trials. This dataset will be fed to the classification
    function.    
    '''
    # Required packages:
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    import pandas as pd
    import numpy as np
    import os
    
    # Making a directory for saving plots:
    davep_folder = os.path.join(plot_path, "DAVEP Analysis")
    try:
        os.mkdir(davep_folder)
    except FileExistsError:
        print("Note: 'DAVEP Analysis' folder already exists.")
        
    # Peak detection and plotting:
    plt.figure(figsize=plot_size)
    peak_locs = list(np.zeros(4))
    plot_names = ['Left Macular', 'Left Peripheral', 'Right Macular', 'Right Peripheral']
    for i in range(4):
        peak_locs[i] = pd.DataFrame(columns=['min', 'ms', 'microV'])
        col_names = all_vs[i].columns 
        for j in range(all_vs[i].shape[1]):
            x = all_vs[i].iloc[:, j] 
            # Detecting peaks:
            peak_ind = find_peaks(x)[0] # You can specify properties of peaks here
            A = pd.DataFrame(np.repeat(col_names[j], len(peak_ind)))
            B = pd.DataFrame(all_vs[i].iloc[peak_ind, j])
            C = pd.DataFrame(B.index)
            D = pd.concat([A.reset_index(drop=True), C.reset_index(drop=True), B.reset_index(drop=True)], axis=1)
            D.columns = ['min', 'ms', 'microV']
            peak_locs[i] = peak_locs[i].append(D, ignore_index=True)
            
        # Plotting and saving the figures:
        plt.subplot(1, 4, i+1)   
        plt.scatter(peak_locs[i].iloc[:, 0], peak_locs[i].iloc[:, 1], marker='.', color='tomato')
        plt.ylim(600, 0)
        plt.title(plot_names[i])
        plt.tight_layout()
    plt.savefig(os.path.join(davep_folder, str(subject_id) + ' - all_peaks.png'), dpi=300, facecolor='w')
    if show_plots == False:
        plt.close()

    return peak_locs
    
    
    






def find_components(peak_locs, plot_path, subject_id, n_lm=4, n_lp=4, n_rm=4, n_rp=4, cl_search='half', plot_size=(15,4), show_plots=True):
    '''
    Author: Ali Banijamali (banijamali.s@northeastern.edu)
    
    Input:
    peak_locs must be a list containing 4 dataframes in this order:
    0: Left Macular Peaks (min, ms, microV)
    1: Left Peripheral Peaks (min, ms, microV)
    2: Right Macular Peaks (min, ms, microV)
    3: Right Peripheral Peaks (min, ms, microV)
    n_lm/lp/rm/rp: Number of clusters for each eye and region. All of these are 4 by default.
    However, by examining the clustering plots these can be changed if necessary.
    plot_size: (width, height)
    plot_path: a folder (DAVEP Analysis), will be created here for
    all results and plots (If it doesn't already exist.)
    show_plots: TRUE/FALSE, whether to show the plots after running or not.
    cl_search: Whether to start clustering the components from 0 ("beginning") or
    from the half the experiment ("half").
    
    Function:
    This function takes the peaks, does a Gaussian Mixture Model classification on
    them to recognize the component and then returns the components centers (min).
    It is designed to get the output of find_trial_peaks function as input.
    
    Output:
    The function saves the clustering results plots in "DAVEP Analysis" folder
    and it returns a list containing 4 arrays in this order:
    0: Left Macular component centers
    1: Left Peripheral component centers
    2: Right Macular component centers
    3: Right Peripheral component centers
    Each one of these dataframes has the location(min) of the components
    This dataset will be fed to another function for detecting the response component.
    '''
    # Required packages:
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    import pandas as pd
    import numpy as np
    import os
    from sklearn.mixture import GaussianMixture as GMM
    
    # Making a directory for saving plots:
    davep_folder = os.path.join(plot_path, "DAVEP Analysis")
    try:
        os.mkdir(davep_folder)
    except FileExistsError:
        print("Note: 'DAVEP Analysis' folder already exists.")
        
    # Classifying and plotting:
    plt.figure(figsize=plot_size)
    centers = list(np.zeros(4))
    plot_names = ['Left Macular', 'Left Peripheral', 'Right Macular', 'Right Peripheral']
    
    if cl_search == 'half':
        cl_start = int(peak_locs[0].shape[0]/2) # Start clustering from half of the experiment
    elif cl_search == 'beginning':
        cl_start = 0 # Start clustering from the start of the experiment
    else:
        raise ValueError("cl_search must be in ['half', 'beginning']")
    
    n_clusters = [n_lm, n_lp, n_rm, n_rp]
    # Clustering the peaks using GMM with diagonal clusters:
    for i in range(4):
        A = np.array(pd.concat([peak_locs[i].iloc[cl_start:, 0], peak_locs[i].iloc[cl_start:, 1]], axis=1))
        gmm = GMM(n_components=n_clusters[i], covariance_type='diag', init_params='kmeans').fit(A)
        labels = gmm.predict(A)
        centers_i = np.zeros(n_clusters[i])
        for j in range(n_clusters[i]):
            centers_i[j] = np.around(A[np.where(labels==j), 1].mean(), 0)
        centers[i] = centers_i

        # Plotting and saving the figures:
        plt.subplot(1, 4, i+1)   
        plt.scatter(A[:, 0], A[:, 1], c=labels, s=40, cmap='viridis')
        plt.ylim(600, 0)
        plt.title(plot_names[i])
        plt.tight_layout()
    plt.savefig(os.path.join(davep_folder, str(subject_id) + ' - all_clusters.png'), dpi=300, facecolor='w')
    if show_plots == False:
        plt.close()

    return centers
    
    
    
    
    
    
    
    
    
def compute_davep_score(all_vs, centers, resp_window_half=75, cs_start=300, cs_end=600, method='normal'):
    '''
    Author: Ali Banijamali (banijamali.s@northeastern.edu)
    
    Input:
    all_vs: Must be a list containing 4 dataframes in this order:
    0: Left Macular Data
    1: Left Peripheral Data
    2: Right Macular Data
    3: Right Peripheral Data
    (!) Depending on method (normal vs. elevated), these can be normal data or elevated to above zero.
    centers: Must be a list containing 4 arrays in this order:
    0: Left Macular component centers (ms)
    1: Left Peripheral component centers (ms)
    2: Right Macular component centers (ms)
    3: Right Peripheral component centers (ms)
    cs_start: component search start at (ms)
    cs_end: component search end at (ms)
    resp_window_half: Half of the window for calculating the score (default=75 i.e. a window of 150 ms)
    The alogorithm looks for the response between these two limits (min). Default: [290, 600] min
    method: 'normal-v1' or 'elevated-v1': Older method for calculating the DAVEP Score
    'normal-v2' or 'elevated-v2': The new method for calculating the DAVEP Score. The score is
    the sum of all peak values withing the response window. The max value is 70 (If all trials are maxed to 1, there are
    70 trials and therefore the max is 70.)
    
    
    Function:
    This function takes the peaks, does a Gaussian Mixture Model classification on
    them to recognize the component and then returns the components centers (min).
    It is designed to get the output of find_trial_peaks function as input.
    
    There are 2 methods for recognizing the response component. Note that only 'P3-Based'
    model works here. The code for the "Length-Based" model only works on the output of Kmeans currently
    and doesn't work on this version. It is here just for the future references.
    'P3-Based': Response cluster is the first cluster after cs_start [ms]
    'Length-Based': Response cluster is the most consistent cluster (longest) of peaks along the experiment [min]
    Length-Based Code:
    # Adding kmeans cluster labels to the peaks locations:
    peak_loc_m_l = pd.concat([peak_loc_m_l, pd.DataFrame(kmeans_m_l.labels_, columns=['label'])], axis=1)
    candid_labels = np.where((centers_m_l[:, 1]>cl_start) & (centers_m_l[:, 1]<cl_end)) # Only considering cluster centers after 175 [ms]
    candids = peak_loc_m_l.loc[peak_loc_m_l['label'].isin(candid_labels[0])] # Candid Clusters info
    candid_count = candids.groupby('label').count() # Counting number of peaks in each of the candid clusters
    response_center_m_l = centers_m_l[candid_count.loc[candid_count['microV'] == candid_count.max().max()].index] # Choosing the cluster w/ highest count of peaks as response
    response_center_m_l = response_center_m_l[0][1]
    
    ! The new score (v2) represents the percentage of trials that have reached the adapted peak value:
    Score = Sum(peak amplitudes of all responses) X 100/number of trials (70)


    
    Output:
    The function saves the clustering results plots in "DAVEP Analysis" folder
    and it returns a list containing 4 arrays in this order:
    0: Left Macular component centers
    1: Left Peripheral component centers
    2: Right Macular component centers
    3: Right Peripheral component centers
    Each one of these dataframes has the location(min) of the components
    
    response_center_info, davep_scores, max_val
    '''
    # Required packages:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    
    # Finding the response and calculating the score:
    response_center_info = np.ndarray((4, 3), dtype=int) # 4: Eye/Regions, 3: Resp Center, Window start, Window end
    davep_scores         = np.zeros(4)
    all_vs = all_vs.copy() 
  
    # Clustering the peaks using GMM with diagonal clusters:
    for i in range(4):
        response_center_info[i, 0] = min(centers[i][(centers[i]>cs_start) & (centers[i]<cs_end)]) # Center of the Response (min)
        response_center_info[i, 1] = int(response_center_info[i, 0]-resp_window_half) # Start of the DAVEP(2) window (Cluster center - 75[ms])
        response_center_info[i, 2] = int(response_center_info[i, 0]+resp_window_half) # End of the DAVEP(2) window (Cluster center + 75 [ms])

    # max_val is the maximum value in the response windows for all 4 eye/regions:
    max_val = max(all_vs[0].iloc[response_center_info[0, 1]:response_center_info[0, 2], :].max().max(),
                  all_vs[1].iloc[response_center_info[1, 1]:response_center_info[1, 2], :].max().max(),
                  all_vs[2].iloc[response_center_info[2, 1]:response_center_info[2, 2], :].max().max(),
                  all_vs[3].iloc[response_center_info[3, 1]:response_center_info[3, 2], :].max().max())
            
    for i in range(4):
        if method == 'normal-v1':
            all_vs[i][all_vs[i]<0] = 0 # Truncating
            all_vs[i] = all_vs[i].iloc[response_center_info[i, 1]:response_center_info[i, 2], :]/max_val # Norm all tVEPs by deviding to the max of the DAVEP WINDOW!
            davep_scores[i] = round(np.linalg.norm(all_vs[i], ord=2), ndigits=1) # Round to 1 decimal
        elif method == 'elevated-v1':
            all_vs[i] = all_vs[i].iloc[response_center_info[i, 1]:response_center_info[i, 2], :]/max_val # Norm all tVEPs by deviding to the max of the DAVEP WINDOW!
            davep_scores[i] = round(np.linalg.norm(all_vs[i], ord=2), ndigits=1) # Round to 1 decimal
        elif method == 'normal-v2':
            all_vs[i][all_vs[i]<0] = 0 # Truncating
            all_vs[i] = all_vs[i].iloc[response_center_info[i, 1]:response_center_info[i, 2], :]/max_val # Norm all tVEPs by deviding to the max of the DAVEP WINDOW!
            davep_scores[i] = round(sum(all_vs[i].max(axis=0))*100/all_vs[0].shape[1], ndigits=1) # Round to 1 decimal
        elif method == 'elevated-v2':
            all_vs[i] = all_vs[i].iloc[response_center_info[i, 1]:response_center_info[i, 2], :]/max_val # Norm all tVEPs by deviding to the max of the DAVEP WINDOW!
            davep_scores[i] = round(sum(all_vs[i].max(axis=0))*100/all_vs[0].shape[1], ndigits=1) # Round to 1 decimal
        
    return response_center_info, davep_scores, max_val
    
    
    
    
    
    
    
    
    
def plot_heatmap(all_vs, response_info, davep_scores, max_val, plot_label, condition, plot_path, subject_id, score_method='normal-v1', sep=10, show_plots=True):
    '''
    Author: Ali Banijamali (banijamali.s@northeastern.edu)
    
    Input:
    all_vs: Must be a list containing 4 dataframes in this order:
    0: Left Macular Data
    1: Left Peripheral Data
    2: Right Macular Data
    3: Right Peripheral Data
    response_info/davep_scores/max_val: These are all the output of find_davep_score function.
    plot_label: String, containing the title of the plot.
    condition: The category of the subject: Health/AMD/...
    score_method: The method by which the score was calculated. This is printed under the plots and
    also the parameters of the heatmap are set based on this value.
    sep: sets the white distance at the center of the heatmap's color bar (the distance between blue and red)
    show_plots: TRUE/FALSE, whether to show the plots after running or not.
    
    
    Function:
    Takes the Vs and other information to plot and save heatmaps.
    
    
    Output:
    Heatmap plots. 
    '''
    
    # Required packages:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os
    
    # Making a directory for saving plots:
    davep_folder = os.path.join(plot_path, "DAVEP Analysis")
    try:
        os.mkdir(davep_folder)
    except FileExistsError:
        print("Note: 'DAVEP Analysis' folder already exists.")
        
    # Setting the heatmap parameters based on score_method:
    hm_params = np.zeros(3)
    if score_method in ['normal-v1', 'normal-v2']:
        hm_params = [-1, 1, 0] # vmin, vmax, center
    elif score_method in ['elevated-v1', 'elevated-v2']:
        hm_params = [0, 1, 0.5] # vmin, vmax, center
        
    # Plotting the title and condition:
    fig = plt.figure(figsize=(6, 4))
    greys = plt.cm.Greys
    ax = fig.add_axes([0, .8, 1, .2])
    ax.text(0, 1, plot_label+'\n\n'+condition, fontsize=20, color=greys(0.8))
    ax.axis('off')
    
    # Plotting the heatmap:
    plot_names = ['Left Macular', 'Left Peripheral', 'Right Macular', 'Right Peripheral']
    x_step = 0.85
    for i in range(3):
        ax = fig.add_axes([i*x_step, 0, .7, .8])
        sns.heatmap(all_vs[i]/max_val, cmap=sns.diverging_palette(258, 11, sep=sep, n=1000, s=99),
                    vmin=hm_params[0], vmax=hm_params[1], center=hm_params[2], cbar=False)
        ax.hlines(y=response_info[i, 0], xmin=0, xmax=all_vs[0].shape[1], linestyles='dashed')
        ax.set_title(plot_names[i]+'\n')
        ax.set_xlabel('Time since Photobleach [min]\n\n' + 'DAVEP Score: ' + str(davep_scores[i]) + '\nMethod: ' + score_method)
        ax.set_ylabel('Time after Stimulus [ms]')
    ax = fig.add_axes([3*x_step, 0, .8, .8]) # 4th plot has a color bar
    sns.heatmap(all_vs[3]/max_val, cmap=sns.diverging_palette(258, 11, sep=sep, n=1000, s=99),
                vmin=hm_params[0], vmax=hm_params[1], center=hm_params[2])
    ax.hlines(y=response_info[3, 0], xmin=0, xmax=all_vs[0].shape[1], linestyles='dashed')
    ax.set_title(plot_names[3]+'\n')
    ax.set_xlabel('Time since Photobleach [min]\n\n' + 'DAVEP Score: ' + str(davep_scores[3]) + '\nMethod: ' + score_method)
    ax.set_ylabel('Time after Stimulus [ms]'); 
    
    
    # Saving the figures:
    plt.savefig(os.path.join(davep_folder, str(subject_id) + ' - heat_map - '+ score_method + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    if show_plots == False:
        plt.close()
        
        
        
        
        
        
        
        
        
def plot_wireframe(all_vs, response_info, davep_scores, max_val, plot_label, condition, plot_path, subject_id, score_method='normal-v1', plot_angles=[70, 30], show_plots=True):
    '''
    Author: Ali Banijamali (banijamali.s@northeastern.edu)
    
    Input:
    all_vs: Must be a list containing 4 dataframes in this order:
    0: Left Macular Data
    1: Left Peripheral Data
    2: Right Macular Data
    3: Right Peripheral Data
    response_info/davep_scores/max_val: These are all the output of find_davep_score function.
    plot_label: String, containing the title of the plot.
    condition: The category of the subject: Health/AMD/...
    score_method: The method by which the score was calculated. This is printed under the plots and
    also the parameters of the heatmap are set based on this value.
    plot_angles: Set the elevation and azimuth of the axes in degrees (not radians). This can be used
    to rotate the axes.
    show_plots: TRUE/FALSE, whether to show the plots after running or not.
    
    
    Function:
    Takes the Vs and other information to plot and save wireframe plots.
    
    
    Output:
    Wireframe plots. 
    '''
    
    # Required packages:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    
    # Making a directory for saving plots:
    davep_folder = os.path.join(plot_path, "DAVEP Analysis")
    try:
        os.mkdir(davep_folder)
    except FileExistsError:
        print("Note: 'DAVEP Analysis' folder already exists.")
        
    # Plotting the title and condition:
    fig = plt.figure(figsize=(6, 4))
    greys = plt.cm.Greys
    ax = fig.add_axes([0, .8, 1, .2])
    ax.text(0, 1, plot_label+'\n\n'+condition, fontsize=20, color=greys(0.8))
    ax.axis('off')
    
    # Plotting the heatmap:
    plot_names = ['Left Macular', 'Left Peripheral', 'Right Macular', 'Right Peripheral']
    X, Y = np.meshgrid(all_vs[0].columns.astype(float), np.arange(0, 601, 1)) # This is done for the wf plots
    YY, ZZ = np.meshgrid(all_vs[0].columns.astype(float), np.arange(0, 1, 0.2)) # This is for reponse center plane
    x_step = 0.8
    for i in range(4):
        ax = fig.add_axes([i*x_step, 0, .9, .8], projection='3d')
        ax.plot_wireframe(X, Y, np.array(all_vs[i]/max_val), rstride=10000)#, color='black')
        ax.view_init(plot_angles[0], plot_angles[1]) # Viewing angle
        ax.invert_xaxis()
        
        # Response center plane:
        #XX = response_info[i, 0]
        #ax.plot_surface(XX, YY, ZZ, alpha=0.2)
        
        ax.set_zlim(0, 1) # Scaling the Z-axis
        ax.set_title(plot_names[i]+'\n')
        ax.set_xlabel('\n\n\nTime since Photobleach [min]\n\n' + 'DAVEP Score: ' + str(davep_scores[i]) + '\nMethod: ' + score_method)
        ax.set_ylabel('Time after Stimulus [ms]')
        ax.set_zlabel('tVEP Amplitude [$\mu$V]')
    
    
    # Saving the figures:
    plt.savefig(os.path.join(davep_folder, str(subject_id) + ' - wire_frame - ' + score_method + '.png'), dpi=300, facecolor='w', bbox_inches='tight')
    if show_plots == False:
        plt.close()
