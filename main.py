# https://pypi.org/project/pipreqs/
# https://pypi.org/project/matlabengine/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/HDD/Programme/MatLab_2022b/bin/glnxa64

matlab_feautre_list_french_data = [
'Sync_CC_selinger',
'Sync_STTC',
'Sync_MI1',
'Sync_MI2',
'Sync_PS',
'Sync_PS_M',
'Sync_Contrast',
'Sync_Contrast_fixed',
'Sync_ISIDistance',
'Sync_SpikeDistance',
'Sync_SpikeSynchronization',
'Sync_ASpikeSynchronization',
'Sync_AISIDistance',
'Sync_ASpikeDistance',
'Sync_RISpikeDistance',
'Sync_RIASpikeDistance',
'Sync_EarthMoversDistance',
'Connectivity_TSPE']

matlab_feautre_list_french_data_test = [
'Sync_RIASpikeDistance',
'Sync_Contrast',
'Connectivity_TSPE'
]

from decorators import get_time
@get_time
def init_function():
    import os
    from sys import platform

    if platform == "linux" or platform == "linux2":
        print("linux")
    elif platform == "darwin":
        print("darwin")
    elif platform == "win32":
        print("win32")
    if os.path.exists(os.path.join(os.getcwd(), "DrCell")) and os.path.isfile(os.path.join(os.getcwd(), "DrCell", "DrCell.m")):
        print("DrCell available")
    else:
        print("DrCell is not available")
        print("Downloading DrCell from GitHub")
        # https://pypi.org/project/directory-downloader/
        from git import Repo
        git_url = "https://github.com/biomemsLAB/DrCell"
        cwd = os.getcwd()
        directory = "DrCell"
        repo_dir = os.path.join(cwd, directory)
        Repo.clone_from(git_url, repo_dir)

@get_time
def get_list_of_files(path, file_type):
    """
        Generates a list with all file path from a given path.
        Parameters
        ----------
        path : string
            Either a path from a single file or a path of a directory.
        file_type : string, list of string
            The extension of the file type which should be taken into account.
            Can be one or more than one extension.

        Returns
        -------
        list_of_files : list of string
            Returns the synchrony of the input spike trains.

        """
    from import_file import getListOfFiles
    import os
    all_files = []
    # check if path is a file or directory
    if os.path.isdir(path):
        all_files = getListOfFiles(path, file_type)
    else:
        all_files.append(path)
    return all_files

@get_time
def calculate_save_matlab_feature(data):
    """
            Calculates, saves and exports the Data in a csv file.
            Parameters
            ----------
            data : list of strings
                List of all files which will be included in the analysis .
            Returns
            -------
            path_of_csv : string
                Returns the path of the csv file where all calculated features and plots are saved.

            """

    from matlab_apy import matlab_calc_feature
    from import_file import choose_import_func
    from export_feature import export_feature_in_csv, export_feature_in_hdf5
    for file_count, file in enumerate(data):
        print(f"Calculating file number {file_count+1} out of {str(len(data))}:")
        spike_list, amp, rec_dur, SaRa = choose_import_func(file)
        for feature_count, feature in enumerate(matlab_feautre_list_french_data_test):
            print(f"Calculating feature number {feature_count+1} out of {str(len(matlab_feautre_list_french_data_test))}:")
            # calculates the features via matlab
            feature_mean, feature_values, feature_std, feature_pref, feature_label = matlab_calc_feature(spike_list, amp, rec_dur, SaRa, feature)
            # saves the feature in csv file, csv_filename is path
            csv_path = export_feature_in_csv(feature, feature_mean, feature_std, feature_values, feature_pref, feature_label, csv_filename="Feature.csv", feature_file=file)
            # saves the feature in hdf5 file
            # export_feature_in_hdf5(feature, feature_mean, feature_std, feature_values, feature_pref, feature_label, af_filename="Feature.hdf5", feature_file=file)
            # plots the feature if necessary

    return csv_path

@get_time
def post_process_feature(csv_path, h5_path=""):
    """
            Manipulates the features if necessary.
            Parameters
            ----------
            data : list of strings
                List of all files which will be included in the analysis .
            Returns
            -------
            path_of_csv : string
                Returns the path of the csv file where all calculated features and plots are saved.

            """

    from manipulate_feature import read_csv_file, apply_DDT_to_CM, get_con_data_frame, get_sync_data_frame
    # from plot_feature import plot_CM, plot_data_over_div_con, plot_data_over_div_sync

    data_frame = read_csv_file(csv_path)
    # read_h5_file(h5_path)
    # manipulates CMs with DDT
    FM = apply_DDT_to_CM(data_frame, faktor_std=2, verbose=True)

    connectivity_data_frame = get_con_data_frame(FM)
    synchrony_data_frame = get_sync_data_frame(data_frame)

    sync_path = "sync_df.json"
    con_path = "con_df.json"

    synchrony_data_frame.to_json(sync_path)
    connectivity_data_frame.to_json(con_path)

    return con_path, sync_path

@get_time
def plot_all_feature(con_json_path, sync_json_path):
    """
            Plots all feature and connectivty matrizes.
            Parameters
            ----------
            data : list of strings
                List of all files which will be included in the analysis .
            Returns
            -------
            path_of_csv : string
                Returns the path of the csv file where all calculated features and plots are saved.

            """
    import pandas as pd
    import numpy as np
    from plot_feature import plot_df_CM, plot_data_over_div_con, plot_data_over_div_sync

    connectivity_data_frame = pd.read_json(con_json_path)
    synchrony_data_frame = pd.read_json(sync_json_path)

    # plot Connectivity Matrix
    for index, row in connectivity_data_frame.iterrows():
        CM = np.array(row.CM)
        CM_DDT = np.array(row.CM_DDT)
        filepath = row.file_name
        # print(f'Plotting CM {index+1} of {connectivity_data_frame.shape[1]}')
        print(f'Plotting CM {index + 1} of {len(connectivity_data_frame.index)}')
        plot_df_CM(CM_DDT, "CM_DDT", filepath)
        plot_df_CM(CM, "CM", filepath)

    # plot sync feature
    from manipulate_feature import sync_list
    for feature in sync_list:
        plot_data_over_div_sync(synchrony_data_frame, feature, mode="seaborn.swarmplot", verbose=True)

    # plot con feature
    from manipulate_feature import con_list
    for feature in con_list:
        plot_data_over_div_con(connectivity_data_frame, feature, verbose=True)


if __name__ == '__main__':

    path = 'D:/FAUbox/Uni/Dr/USA/Data/MEA Data/'
    init_function()
    print("Import of data ...")

    all_h5_files = get_list_of_files(path, [".csv"])
    # print(all_h5_files)
    print("Total number of files: " + str(len(all_h5_files)))

    csv_feature_path = calculate_save_matlab_feature(all_h5_files)
    csv_feature_path = "Feature.csv"
    print("Feature Calculation completly done")

    con_json_path, sync_json_path = post_process_feature(csv_feature_path)
    print("Post Processing of Connectivty Matrix done")

    con_json_path = "con_df.json"
    sync_json_path = "sync_df.json"

    plot_all_feature(con_json_path, sync_json_path)
    print("Ploting of Feature and Connectivty Matrix done")

    print("Finished complete Analyses of Data")









