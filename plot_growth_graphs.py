# Graphical assessment of a child's development
# according to the World Health Organization growth standards

# Repository & documentation:
# http://github.com/dqsis/child-growth-charts
# -------------------------------------


# Import libraries
import os
import json
import sys
from sys import exit
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from io import BytesIO
import io
import zipfile
import argparse


def unzip_file_to_dir(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

def sort_based_on_first_col(array):
    sorted_idxs = np.argsort(array[:, 0])
    return array[sorted_idxs, :]

def plot_growth(world_health_array, measurements, num_of_displayed_months, y_label, x_label, output_file=None):
    # Plots
    # plt.figure()
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig, ax = plt.subplots()
    percentiles = ['2%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '98%']

    # Age vs weight
    for i in range(1, 10):
        linetype = 'c-' if i == 5 else 'c--'
        if num_of_displayed_months is None:
            x = world_health_array[:, 0]
            y = world_health_array[:, i]
        else:
            x = world_health_array[:num_of_displayed_months + 1, 0]
            y = world_health_array[:num_of_displayed_months + 1, i]
        ax.plot(x, y, linetype, linewidth=0.8)

    ax.plot(measurements[:, 0], measurements[:, 1], 'b-')


    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if num_of_displayed_months is not None:
        ax.set_xlim([0, num_of_displayed_months])
        ax.set_xticks(np.arange(0, num_of_displayed_months + 1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    # We change the fontsize of minor ticks label
    ax.tick_params(axis='both', which='major', labelsize=5)
    # ax.tick_params(axis='both', which='minor', labelsize=3)
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='0.8', linestyle='--')

    for i in range(1, 10):
        if num_of_displayed_months is None:
            x = world_health_array[-1, 0]
            y = world_health_array[-1, i]
        else:
            x = world_health_array[num_of_displayed_months, 0]
            y = world_health_array[num_of_displayed_months, i]

        ax.text(x, y, percentiles[i-1], fontsize=7)
    fig = plt.gcf()
    plt.show()

    if output_file:
        # plt.show()
        plt.draw()
        fig.set_size_inches(16, 9)
        fig.savefig(output_file, dpi=100)


def get_interpolated_value(time_val, times, values):
    """

    :param time_val: query value
    :param times: all times in which the value has been measured
    :param values: the values corresponding to the measurements
    :return:
    """

    num_values = len(values)
    assert time_val >= np.min(times)
    assert time_val <= np.max(times)


    sorted_idxs = np.argsort(times)
    sorted_times = times[sorted_idxs]
    sorted_values = values[sorted_idxs]

    idx = np.sum(sorted_times <= time_val) - 1
    higher_idx = idx + 1

    lower_time = sorted_times[idx]
    lower_value = sorted_values[idx]

    if higher_idx > num_values - 1:  # if at the end of the vector
        return lower_value, idx
    else:
        higher_time = sorted_times[higher_idx]
        higher_value = sorted_values[higher_idx]

        a = (time_val - lower_time)/(higher_time - lower_time)

        assert 0 <= a <= 1

        interpolated_value = lower_value * (1 - a) + higher_value * a
        return interpolated_value, idx

def get_multiple_interpolated_outputs(time_val_vector, times, values):
    out_vector = []
    for time in time_val_vector:
        out, _ = get_interpolated_value(time, times, values)
        out_vector.append(out)
    out_vector = np.array(out_vector)

    return out_vector



parser = argparse.ArgumentParser(description="Default parser for creating tasks")
parser.add_argument('--baby_tracker_abt_file', '-f', default=None, type=str,
                    help='Path of the baby tracker exported file')
parser.add_argument('--num_months', '-m', default=None, type=int,
                    help='Num of months to display')

if __name__ == "__main__":
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(this_dir, 'data')
    output_dir = os.path.join(this_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    file_dir = os.path.dirname(args.baby_tracker_abt_file)
    unzip_file_to_dir(args.baby_tracker_abt_file, output_dir)

    baby_json_path = os.path.join(output_dir, 'baby.json')
    baby_dict = json.load(open(baby_json_path))

    birthdate = baby_dict['birthday']
    datetime_birthdate_object = datetime.strptime(birthdate, '%Y-%m-%d')

    gender = None
    if baby_dict['gender'] in ['BOY', 'boy', 'male']:
        gender = 'b'  # boy
    elif baby_dict['gender'] in ['GIRL', 'girl', 'female']:
        gender = 'g'  # girl
    else:
        print("Error")
        sys.exit(1)


    height_measurements = []
    head_circumference_measurements = []
    weight_measurements = []

    for record in baby_dict['records']:
        type = record['type']
        subtype = record['subtype']
        datetime_str = record['fromDate']
        datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        time_diff = datetime_object - datetime_birthdate_object
        time_in_months = time_diff.days / (365/12)
        # print(type(datetime_object))
        # print(datetime_object)  # printed in default format
        amount = record['amount']

        if type == 'GROWTH':
            if subtype == 'GROWTH_HEIGHT':
                height_measurements.append([time_in_months, amount])
            if subtype == 'GROWTH_WEIGHT':
                weight_measurements.append([time_in_months, amount])
            if subtype == 'GROWTH_HEAD':
                head_circumference_measurements.append([time_in_months, amount])

    height_measurements = sort_based_on_first_col(np.array(height_measurements))
    weight_measurements = sort_based_on_first_col(np.array(weight_measurements))
    head_circumference_measurements = sort_based_on_first_col(np.array(head_circumference_measurements))

    max_time = min(height_measurements[:, 0].max(), weight_measurements[:, 0].max(), head_circumference_measurements[:, 0].max())

    # get height vs weight measurements
    t = np.linspace(0, max_time, 100)
    height_interpol_measurements = get_multiple_interpolated_outputs(time_val_vector=t, times=height_measurements[:, 0], values=height_measurements[:, 1])
    weight_interpol_measurements = get_multiple_interpolated_outputs(time_val_vector=t, times=weight_measurements[:, 0], values=weight_measurements[:, 1])

    height_weight_interpol_measurements = np.stack([height_interpol_measurements, weight_interpol_measurements], axis=1)
    # out3 = get_multiple_interpolated_outputs(time_val_vector=t, times=head_circumference_measurements[:, 0], values=head_circumference_measurements[:, 1])
    # plt.figure()
    # plt.plot(t, out1)
    # plt.plot(t, out2)
    # plt.plot(t, out3)
    # plt.show()



    # percentiles = ['2%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '98%']

    # Read age vs weight WHO data
    age_weight_file = f'{gender}_age_weight.csv'
    age_length_file = f'{gender}_age_length.csv'
    age_headc_file = f'{gender}_age_headc.csv'
    length_weight_file = f'{gender}_length_weight.csv'

    age_weight_array = np.loadtxt(os.path.join(data_path, age_weight_file), delimiter=',', skiprows=1)
    age_length_array = np.loadtxt(os.path.join(data_path, age_length_file), delimiter=',', skiprows=1)
    age_headc_array = np.loadtxt(os.path.join(data_path, age_headc_file), delimiter=',', skiprows=1)
    length_weight_array = np.loadtxt(os.path.join(data_path, length_weight_file), delimiter=',', skiprows=1)

    """ Num of displayed month is 1 less than the number of records (month 0 doesn't count)"""
    num_of_displayed_months = age_weight_array.shape[0] - 1

    if args.num_months is not None:
        num_of_displayed_months = min(args.num_months, num_of_displayed_months)

    world_health_array = age_weight_array
    measurements = weight_measurements

    # Age vs weight
    plot_growth(
        world_health_array=age_weight_array,
        measurements=weight_measurements,
        num_of_displayed_months=num_of_displayed_months,
        y_label='weight [kg]',
        x_label='age [months]',
        output_file=os.path.join(output_dir, 'age_weight_growth.png')
    )

    # Age vs length
    plot_growth(
        world_health_array=age_length_array,
        measurements=height_measurements,
        num_of_displayed_months=num_of_displayed_months,
        y_label='length [cm]',
        x_label='age [months]',
        output_file=os.path.join(output_dir, 'age_height_growth.png')
    )

    # Age vs head circumference
    plot_growth(
        world_health_array=age_headc_array,
        measurements=head_circumference_measurements,
        num_of_displayed_months=num_of_displayed_months,
        y_label='length [cm]',
        x_label='age [months]',
        output_file=os.path.join(output_dir, 'age_head_circ_growth.png')
    )

    max_height = height_weight_interpol_measurements[:, 0].max()
    _, num_of_datapoints = get_interpolated_value(max_height, length_weight_array[:, 0], length_weight_array[:, 0])

    # weight vs height
    plot_growth(
        world_health_array=length_weight_array,
        measurements=height_weight_interpol_measurements,
        num_of_displayed_months=None, #int(num_of_datapoints) + 1,
        y_label='weight [kg]',
        x_label='length [cm]',
        output_file=os.path.join(output_dir, 'weight_length_growth.png')
    )



    # # Length vs weight
    # plt.subplot(2,2,4)
    # plt.plot(\
    # lwarray[:,0],lwarray[:,1],'r--',\
    # lwarray[:,0],lwarray[:,2],'r--',\
    # lwarray[:,0],lwarray[:,3],'r--',\
    # lwarray[:,0],lwarray[:,4],'r--',\
    # lwarray[:,0],lwarray[:,5],'k-',\
    # lwarray[:,0],lwarray[:,6],'r--',\
    # lwarray[:,0],lwarray[:,7],'r--',\
    # lwarray[:,0],lwarray[:,8],'r--',\
    # lwarray[:,0],lwarray[:,9],'r--',\
    # chlwarrayX,chlwarrayY,'b-*')
    #
    # plt.grid(True)
    # plt.xlabel('lenght [cm]')
    # plt.ylabel('weight [kg]')
    # plt.xlim([45,110])
    # plt.xticks(np.arange(45,111,10))
    #
    # plt.text(lwarray[73,0], lwarray[73,1],'2%',fontsize=7)
    # plt.text(lwarray[80,0], lwarray[80,2],'5%',fontsize=7)
    # plt.text(lwarray[87,0], lwarray[87,3],'10%',fontsize=7)
    # plt.text(lwarray[94,0], lwarray[94,4],'25%',fontsize=7)
    # plt.text(lwarray[101,0], lwarray[101,5],'50%',fontsize=7)
    # plt.text(lwarray[108,0], lwarray[108,6],'75%',fontsize=7)
    # plt.text(lwarray[115,0], lwarray[115,7],'90%',fontsize=7)
    # plt.text(lwarray[122,0], lwarray[122,8],'95%',fontsize=7)
    # plt.text(lwarray[125,0], lwarray[125,9],'98%',fontsize=7)

    # # Adjust distance between subplots
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    #
    # # Show & save graphs
    # fig1 = plt.gcf()
    # plt.show()
    # plt.draw()
    #
    # fig1.set_size_inches(16, 9)
    # fig1.savefig(os.path.join(file_dir, 'growth.pdf'), dpi=100)
    # fig1.savefig(os.path.join(file_dir, 'growth.png'), dpi=100)

