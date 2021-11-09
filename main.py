import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import streamlit as st

# assumes standard SSD, field sizes, obliquity
###################

fs_dictionary = {'4x4': 1, '6x6': 2, '6x10': 3, '10x10': 4, '15x15': 5, '20x20': 6, '25x25': 7}
energy_dictionary = {0: '6 MeV', 1: '9 MeV', 2: '12 MeV', 3: '16 MeV', 4: '20 MeV'}
data = np.load('pdds.npy', allow_pickle=True)

Rs = np.zeros((5, 8, 5), dtype='float32')
for i in range(5):
    for j in range(8):
        if j == 0:
            continue
        interpolant = interp1d(data[i][:, 0], data[i][:, j])
        fine_res_depths = np.linspace(0, data[i][-1, 0], 1000)
        fine_res_dose = interpolant(fine_res_depths)
        d_max_idx = np.where(np.abs(fine_res_dose - 100) == np.amin(np.abs(fine_res_dose - 100)))[0][0]
        ranges = [90, 85, 80, 50]
        for r_idx in range(len(ranges) + 1):
            if r_idx == 0:
                Rs[i, j, r_idx] = fine_res_depths[d_max_idx]
                continue
            d_idx = np.where(np.abs(fine_res_dose[d_max_idx:] - ranges[r_idx - 1]) == np.amin(np.abs(fine_res_dose[d_max_idx:] - ranges[r_idx - 1])))[0]
            d_idx += d_max_idx
            Rs[i, j, r_idx] = fine_res_depths[d_idx]
R90s = Rs[:, :, 0]


def print_out(t_min, t_max, field_size, energy_index, bolus_thickness, oar_depth=None, plot=False, verbose=True,
              Rx_vol=95, Rx_dose=100, norm_method='vol'):
    if oar_depth is not None:
        oar_depth = 1.5 * t_max
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
    d_oar_idx = np.where(np.abs(fine_res_depths - (oar_depth + bolus_thickness)) == np.amin(
        np.abs(fine_res_depths - (oar_depth + bolus_thickness))))[0][0]
    d_oar = fine_res_dose[d_oar_idx]
    d_max_dose = np.amax(fine_res_dose)

    line_doses = fine_res_dose[t_min_idx:t_max_idx]
    if norm_method == 'vol':
        pRx = np.zeros(line_doses.shape)
        for didx in range(line_doses.shape[0]):
            norm_test = (line_doses / line_doses[didx]) * 100
            vols = norm_test > Rx_dose
            pRx[didx] = vols.astype('float').sum() / vols.shape[0]
        prx_vol  = Rx_vol / 100
        where_vol = np.where(np.abs(pRx - prx_vol) == np.amin(np.abs(pRx - prx_vol)))[0][0]
        norm_val = line_doses[where_vol]
    elif norm_method == 't_min':
        norm_val = t_min_dose
    elif norm_method == 't_max':
        norm_val = t_max_dose

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

    t_min_dose /= (norm_val / 100)
    d_max_dose /= (norm_val / 100)
    skin_dose /= (norm_val / 100)
    d_oar /= (norm_val / 100)
    t_max_dose /= (norm_val / 100)

    if verbose:
        print('Energy: ', energy_dictionary[energy_index], ', Bolus: {} mm'.format(bolus_thickness), \
              ', t_min Dose: {}%'.format(np.round(t_min_dose, 2)), ', t_max Dose: {}%'.format(np.round(t_max_dose, 2)), \
              ', Hot Spot Dose: {}%'.format(np.round(d_max_dose, 2)), ', Skin Dose: {}%'.format(np.round(skin_dose, 2)), \
              ', D(OAR): {}%'.format(np.round(d_oar, 2)))

    return [t_min_dose, t_max_dose, d_max_dose, skin_dose, d_oar, norm_val]


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

    # print_out(t_min, t_max, field_size, recommended_energy_index, recommended_bolus, plot=False)

    return recommended_energy, recommended_bolus


def brute_force(t_min, t_max, field_size, oar_depth, oar_target_dose=50, w_t_min=1., w_t_max=1., w_hotspot=1., w_skin=1.,
                w_depth=1., norm_method='vol', norm_dose=100, norm_vol=95):
    possibilities = []
    depth_dose = np.round(oar_depth, 2)
    for energy_index in range(5):
        for bolus_thickness in [0, 3, 5, 8, 10, 13]:
            po = print_out(t_min, t_max, field_size, energy_index, bolus_thickness, oar_depth_input, verbose=False,
                           norm_method=norm_method, Rx_dose=norm_dose, Rx_vol=norm_vol)
            skin_err = w_skin * (max(90, po[3]) - 90)
            depth_err = w_depth * (max(oar_target_dose, po[4]) - oar_target_dose)
            hotspot_error = w_hotspot * (po[2] - 100)
            t_min_error = w_t_min * np.abs(100 - po[0])
            t_max_error = w_t_max * np.abs(100 - po[1])

            # print(energy_dictionary[energy_index], bolus_thickness)
            # print('skin', skin_err)
            # print('depth', depth_err)
            # print('t_min', t_min_error)
            # print('hot_spot', hotspot_error)

            error = t_min_error + hotspot_error + skin_err + depth_err + t_max_error
            possibilities.append(
                [energy_dictionary[energy_index] + ', ' + str(bolus_thickness) + ' mm'] + po + [error])

    possibilities = pd.DataFrame(np.array(possibilities), columns=['Energy, Bolus', 'Target Entrance Dose',
                                                                   'Target Exit Dose', 'Hot Spot', 'Skin Dose',
                                                                   'Depth Dose ({} mm)'.format(depth_dose), 'norm', 'Error'])
    possibilities['Skin Dose'] = possibilities['Skin Dose'].astype('float').round(2)
    possibilities['Target Exit Dose'] = possibilities['Target Exit Dose'].astype('float').round(2)
    possibilities['Target Entrance Dose'] = possibilities['Target Entrance Dose'].astype('float').round(2)
    possibilities['Hot Spot'] = possibilities['Hot Spot'].astype('float').round(2)
    possibilities['Error'] = possibilities['Error'].astype('float')
    possibilities['norm'] = possibilities['norm'].astype('float')
    possibilities['Depth Dose ({} mm)'.format(depth_dose)] = possibilities[
        'Depth Dose ({} mm)'.format(depth_dose)].astype('float').round(2)

    possibilities = possibilities.sort_values(by=['Error'], axis=0, ascending=True)

    return possibilities

st.set_page_config(layout='wide')
st.title('Electron Energy/Bolus Estimator')
input_cols = st.columns((1, 4, 4, 4, 1, 3, 1))
t_min_input = input_cols[1].number_input('Target Min Depth [cm]', value=1., step=0.05)
t_max_input = input_cols[2].number_input('Target Max Depth [cm]', value=2., step=0.05)
field_size_input = input_cols[3].selectbox('Cone Size: ', ('4x4', '6x6', '6x10', '10x10', '15x15', '20x20', '25x25'),
                                           index=3)
fs_idx = fs_dictionary[field_size_input]
input_norm = input_cols[5].radio('Normalization: ', ('100% dose to 95% of volume', '100% to max depth', '100% to min depth'))
input_norm_dose = 0
input_norm_vol = 0
if input_norm == '100% to max depth':
    input_norm_method = 't_max'
elif input_norm == '100% to min depth':
    input_norm_method = 't_min'
elif input_norm == '100% dose to 95% of volume':
    input_norm_method = 'vol'
    input_norm_vol = 95
    input_norm_dose = 100

if t_max_input < t_min_input:
    st.error('Target min depth should be < target max depth.')
    quit()

with st.expander("Advanced"):
    advanced_cols1 = st.columns(2)
    oar_depth_input = advanced_cols1[0].number_input('OAR Depth [cm]', value=1.5 * t_max_input, step=0.05)
    oar_target_dose_input = advanced_cols1[1].number_input('OAR Target Dose [%Rx Dose]', value=30., step=1.)
    advanced_cols2 = st.columns(4)
    w_t_input = advanced_cols2[0].number_input('Entrance Dose Coverage Priority', value=1., step=0.1)
    w_hotspot_input = advanced_cols2[1].number_input('Hotspot Reduction Priority', value=1., step=0.1)
    w_skin_input = advanced_cols2[2].number_input('Skin Dose Reduction Priority', value=1., step=0.1)
    w_depth_input = advanced_cols2[3].number_input('OAR Sparing Priority', value=1., step=0.1)

t_min_input *= 10
t_max_input *= 10
oar_depth_input *= 10

output = brute_force(t_min_input, t_max_input, field_size_input, oar_depth_input, oar_target_dose_input,
                     w_t_min=w_t_input, w_hotspot=w_hotspot_input, w_depth=w_depth_input, w_skin=w_skin_input,
                     norm_method=input_norm_method, norm_dose=input_norm_dose, norm_vol=input_norm_vol)
output = output.set_index(['Energy, Bolus'])

filter_col = output.to_numpy()[:, -3]
where_stop = np.where(filter_col < 90)[0] # depth dose must be < 90
filtered = output.iloc[where_stop]
filter_col2 = filtered.to_numpy()[:, -6]
where_stop2 = np.where(filter_col2 > 60)  # target entrance dose must be > 60
filtered = filtered.iloc[where_stop2]
filter_col3 = filtered.to_numpy()[:, -5]
where_stop3 = np.where(filter_col3 < 150)  # hot spot dose must be < 150
filtered = filtered.iloc[where_stop3]

filtered = filtered.drop(columns=['norm', 'Error'])
out_cols = st.columns((1, 4, 1))
if filtered.empty:
    out_cols[1].write('No acceptable plans were found. Please verify your inputs.')
    out_cols[1].write('')
    out_cols[1].write('Would you like to display possibilities?')
    if out_cols[1].button('Proceed'):
        out_cols[1].dataframe(output.drop(columns=['norm', 'Error']), width=2000, height=1000)
else:
    out_cols[1].dataframe(filtered, width=2000, height=1000)

    colors = ['royalblue', 'darkgoldenrod', 'green', 'darkred', 'coral', 'orchid', 'lightgreen', 'navy']
    boluses = [0, 3, 5, 8, 10, 13]
    im_cols = st.columns((1, 3, 3, 1))
    for bt in range(6):
        fig, axs = plt.subplots()
        for e in range(5):
            interpolant = interp1d(data[e][:, 0], data[e][:, fs_idx])
            fine_res_depths = np.linspace(0, data[e][-1, 0], 1000)
            fine_res_dose = interpolant(fine_res_depths)
            t_max_idx = np.where(np.abs(fine_res_depths - (t_max_input + boluses[bt])) == np.amin(
                np.abs(fine_res_depths - (t_max_input + boluses[bt]))))[0][0]
            t_max_dose = fine_res_dose[t_max_idx]
            d_oar_plot_idx = np.where(np.abs(fine_res_depths - (2. * t_max_input + boluses[bt])) == np.amin(
                np.abs(fine_res_depths - (2. * t_max_input + boluses[bt]))))[0][0]
            d_oar_plot = fine_res_dose[d_oar_plot_idx]
            d_oar_plot /= t_max_dose
            try:
                norm_value = output['norm'].loc[energy_dictionary[e] + ', ' + str(boluses[bt]) + ' mm']
            except:
                continue
            rx_normed = (data[e][:, fs_idx] / norm_value) * 100
            if np.amax(rx_normed) > 150 or d_oar_plot > 0.85:
                continue
            axs.plot(data[e][:, 0], rx_normed, label='{} PDD'.format(energy_dictionary[e]), color=colors[e])
        axs.plot([t_min_input + boluses[bt], t_min_input + boluses[bt]], [0, 150], label='t_min + bolus', color=colors[-1])
        axs.plot([t_max_input + boluses[bt], t_max_input + boluses[bt]], [0, 150], label='t_max + bolus', color=colors[-2])
        if boluses[bt] != 0:
            axs.plot([boluses[bt], boluses[bt]], [0, 150], label='skin surface', color=colors[-3])
        axs.set_xlim([0, np.minimum(3 * t_max_input, data[e][-1, 0])])
        axs.set_ylim([0, 150])
    
        axs.set_title('{} mm bolus'.format(boluses[bt]))
        axs.legend(fontsize='x-small', loc='upper right')
        axs.set_xlabel('Depth [mm]')
        axs.set_ylabel('Percent Rx Dose')
        fig.savefig('{}PDD.png'.format(boluses[bt]))
        col_idx = bt % 2 + 1
        im_cols[col_idx].image('{}PDD.png'.format(boluses[bt]))

st.markdown('***')
st.markdown('### Ranges for a {} cone (mm):'.format(field_size_input))
range_cols = st.columns((1, 3, 1))
range_display = Rs[:, fs_idx]
range_display = pd.DataFrame(range_display,
                             index=['6 MeV', '9 MeV', '12 MeV', '16 MeV', '20 MeV'],
                             columns=['Rmax', 'R90', 'R85', 'R80', 'R50']
                             ).astype('float').round(2)
range_cols[1].table(range_display)
