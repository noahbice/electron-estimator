import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import streamlit as st

# assumes standard SSD, field sizes, obliquity
###################
# inputs
t_min = 15
t_max = 40
field_size = '10x10'
plot_pdds = False
##################

fs_dictionary = {'4x4': 1, '6x6': 2, '6x10': 3, '10x10': 4, '15x15': 5, '20x20': 6, '25x25': 7}
energy_dictionary = {0: '6 MeV', 1:'9 MeV', 2: '12 MeV', 3:'16 MeV', 4:'20 MeV'}
fs_idx = fs_dictionary[field_size]
data = np.load('pdds.npy', allow_pickle=True)

if plot_pdds:
    for i in range(5):
        plt.plot(data[i][:, 0], data[i][:, fs_idx], label=energy_dictionary[i])
    plt.legend()
    plt.xlabel('Depth', fontsize=14)
    plt.ylabel('Percent max dose', fontsize=14)
    plt.title('PDDs for a {} field'.format(field_size))
    plt.show()

R90s = np.zeros((5, 8), dtype='float32')
for i in range(5):
    for j in range(8):
        if j == 0:
            R90s[i, j] = 0
            continue
        interpolant = interp1d(data[i][:, 0], data[i][:, j])
        fine_res_depths = np.linspace(0, data[i][-1, 0], 1000)
        fine_res_dose = interpolant(fine_res_depths)
        d_max_idx = np.where(np.abs(fine_res_dose - 100) == np.amin(np.abs(fine_res_dose - 100)))[0][0]
        d_idx = np.where(np.abs(fine_res_dose[d_max_idx:] - 90) == np.amin(np.abs(fine_res_dose[d_max_idx:] - 90)))[0]
        d_idx += d_max_idx
        R90s[i, j] = fine_res_depths[d_idx]

def print_out(t_min, t_max, field_size, energy_index, bolus_thickness, plot=False, verbose=True):
    fs_idx = fs_dictionary[field_size]

    interpolant = interp1d(data[energy_index][:, 0], data[energy_index][:, fs_idx])
    fine_res_depths = np.linspace(0, data[energy_index][-1, 0], 1000)
    fine_res_dose = interpolant(fine_res_depths)
    t_min_idx = np.where(np.abs(fine_res_depths - (t_min + bolus_thickness)) == np.amin(
        np.abs(fine_res_depths - (t_min + bolus_thickness))))[0][0]
    t_min_dose = fine_res_dose[t_min_idx]
    t_max_idx = np.where(np.abs(fine_res_depths - (t_max + bolus_thickness)) == np.amin(
        np.abs(fine_res_depths - (t_max + bolus_thickness))))[0][0]
    t_max_dose = fine_res_dose[t_max_idx]
    skin_idx = np.where(np.abs(fine_res_depths - bolus_thickness) == np.amin(
        np.abs(fine_res_depths - bolus_thickness)))[0][0]
    skin_dose = fine_res_dose[skin_idx]
    d_1pt5tmax_idx = np.where(np.abs(fine_res_depths - (1.5 * t_max + bolus_thickness)) == np.amin(
        np.abs(fine_res_depths - (1.5 * t_max + bolus_thickness))))[0][0]
    d_1pt5tmax = fine_res_dose[d_1pt5tmax_idx]
    d_max_dose = np.amax(fine_res_dose)

    if plot == True:
        Rx_normed = (data[energy_index][:, fs_idx] / t_max_dose) * 100
        plt.plot(data[energy_index][:, 0], Rx_normed, label='{} PDD'.format(energy_dictionary[energy_index]))
        plt.plot([t_min + bolus_thickness, t_min + bolus_thickness], [0, 100], label='t_min + bolus')
        plt.plot([t_max + bolus_thickness, t_max + bolus_thickness], [0, 100], label='t_max + bolus')
        plt.xlim([0, data[energy_index][-1, 0]])
        plt.ylim([0, np.amax(Rx_normed) + 0.05])
        plt.plot([bolus_thickness, bolus_thickness], [0, 100], label='skin surface')
        plt.title('{} PDD  with {} mm bolus'.format(bolus_thickness, bolus_thickness))
        plt.legend()
        plt.xlabel('Depth', fontsize=14)
        plt.ylabel('Percent Rx Dose', fontsize=14)
        plt.show()

    # Renormalization
    t_min_dose /= (t_max_dose / 100)
    d_max_dose /= (t_max_dose / 100)
    skin_dose /= (t_max_dose / 100)
    d_1pt5tmax /= (t_max_dose / 100)

    t_max_dose /= (t_max_dose / 100)

    if verbose:
        print('Energy: ', energy_dictionary[energy_index], ', Bolus: {} mm'.format(bolus_thickness), \
              ', t_min Dose: {}%'.format(np.round(t_min_dose, 2)), ', t_max Dose: {}%'.format(np.round(t_max_dose, 2)), \
              ', Hot Spot Dose: {}%'.format(np.round(d_max_dose, 2)), ', Skin Dose: {}%'.format(np.round(skin_dose, 2)), \
              ', D(1.5 t_max): {}%'.format(np.round(d_1pt5tmax, 2)))

    return [t_min_dose, t_max_dose, d_max_dose, skin_dose, d_1pt5tmax]

def sunshine_logic(t_min, t_max, field_size):
    fs_idx = fs_dictionary[field_size]

    # Select energy to match R90
    for i in range(5):
        # print(R90s[i, fs_idx], t_max)
        if R90s[i, fs_idx] < t_max:
            continue
        else:
            recommended_energy = energy_dictionary[i]
            recommended_energy_index = i
            break

    # Select bolus if "overshooting"
    recommended_bolus = 0
    if np.abs(R90s[recommended_energy_index, fs_idx] - t_max) > 10:
        recommended_bolus += 10
    elif np.abs(R90s[recommended_energy_index, fs_idx] - t_max) > 5:
        recommended_bolus += 5
    elif np.abs(R90s[recommended_energy_index, fs_idx] - t_max) > 3:
        recommended_bolus += 3

    print_out(t_min, t_max, field_size, recommended_energy_index, recommended_bolus, plot=True)

    return recommended_energy, recommended_bolus

def brute_force(t_min, t_max, field_size, w1=1., w2=1., w3=1., w4=1.):
    possibilities = []
    depth_dose = np.round(1.5 * t_max, 2)
    for energy_index in range(5):
        for bolus_thickness in [3, 5, 10]:
            po = print_out(t_min, t_max, field_size, energy_index, bolus_thickness, verbose=False)
            error = (w1 * (100 - po[0]) ** 2) + (w2 * (100 - po[2]) **2) + (w3 * po[3] ** 2) + (w4 * (po[4]) ** 2)
            possibilities.append([energy_dictionary[energy_index], bolus_thickness] + po + [error] )
    possibilities = pd.DataFrame(np.array(possibilities), columns=['Energy', 'Bolus', 'Target Entrance Dose',
                                                                   'Target Exit Dose', 'Hot Spot', 'Skin Dose',
                                                                   'Depth Dose ({} mm)'.format(depth_dose), 'Error'])
    possibilities['Skin Dose'] = possibilities['Skin Dose'].astype('float').round(2)
    possibilities['Target Exit Dose'] = possibilities['Target Exit Dose'].astype('float').round(2)
    possibilities['Target Entrance Dose'] = possibilities['Target Entrance Dose'].astype('float').round(2)
    possibilities['Hot Spot'] = possibilities['Hot Spot'].astype('float').round(2)
    possibilities['Depth Dose ({} mm)'.format(depth_dose)] = possibilities['Depth Dose ({} mm)'.format(depth_dose).format(np.round(1.5*t_max, 2))].astype('float').round(2)

    possibilities = possibilities.sort_values(by=['Error'], axis=0, ascending=True)
    # possibilities = possibilities.drop(columns=['Error'])
    
    return possibilities

# sunshine_logic(t_min, t_max, field_size)
brute_force(t_min, t_max, field_size)

st.set_page_config(layout='wide')
st.title('Electron Energy/Bolus Estimator')
t_min_input = st.number_input('Target Min Depth [mm]', value=10.)
t_max_input = st.number_input('Target Max Depth [mm]', value=20.)
field_size_input = st.selectbox('Field Size: ', ('4x4', '6x6', '6x10', '10x10', '15x15', '20x20', '25x25'), index=3)
output = brute_force(t_min_input, t_max_input, field_size_input)
# st.dataframe(output, width=2000, height=1000)
st.table(output)

boluses = [3, 5, 10]
for bt in range(3):
    fig, axs = plt.subplots()
    for e in range(5):
        interpolant = interp1d(data[e][:, 0], data[e][:, fs_idx])
        fine_res_depths = np.linspace(0, data[e][-1, 0], 1000)
        fine_res_dose = interpolant(fine_res_depths)
        t_max_idx = np.where(np.abs(fine_res_depths - (t_max + boluses[bt])) == np.amin(np.abs(fine_res_depths - (t_max + boluses[bt]))))[0][0]
        t_max_dose = fine_res_dose[t_max_idx]
        
        Rx_normed = (data[e][:, fs_idx] / t_max_dose) * 100
        axs.plot(data[e][:, 0], Rx_normed, label='{} PDD'.format(energy_dictionary[e]))
    axs.plot([t_min + boluses[bt], t_min + boluses[bt]], [0, 100], label='t_min + bolus')
    axs.plot([t_max + boluses[bt], t_max + boluses[bt]], [0, 100], label='t_max + bolus')
    axs.plot([boluses[bt], boluses[bt]], [0, 100], label='skin surface')
    axs.set_xlim([0, data[e][-1, 0]])
    axs.set_ylim([0, 150])
        
    axs.set_title('{} mm bolus'.format(boluses[bt]))
    axs.legend(fontsize='xsmall', loc='upper right')
    axs.set_xlabel('Depth')
    axs.set_ylabel('Percent Rx Dose')
    fig.savefig('{}PDD.png'.format(boluses[bt]))
    st.image('{}PDD.png'.format(boluses[bt]))

