import os, datetime
import spym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


from pathlib import Path
from xarray import DataArray
from enum import Enum

class SM4File:
    FileType = Enum("FileType", ["Image", "dIdV", "IZ"])
    def __init__(self, src_path: str):
        self.path = Path(src_path)
        self.fname = self.path.stem
        self.file = spym.load(src_path)
        self.fig = None
        self.type = None

        if 'Current' in self.file.data_vars:
            match self.file.Current.RHK_LineTypeName:
                case 'RHK_LINE_IV_SPECTRUM':
                    self.type = SM4File.FileType.dIdV
                case 'RHK_LINE_IZ_SPECTRUM':
                    self.type = SM4File.FileType.IZ
        elif 'Topography_Forward' in self.file.data_vars:
            self.type = SM4File.FileType.Image

        if self.file is not None:
            self.parse_data(self.type)

    def parse_data(self, T: FileType):
        match T:
            case SM4File.FileType.Image:
                self._topography = self.file.Topography_Forward
            case SM4File.FileType.dIdV:
                self._spec = self.file.LIA_Current
            case SM4File.FileType.IZ:
                self._spec = self.file.Current
            case _:
                pass
        
    def generate_preview(self):
        match self.type:
            case SM4File.FileType.Image:
                self.image_preview(self._topography)
            case SM4File.FileType.dIdV:
                if len(self._spec.LIA_Current_y) > 0:
                    try:
                        self.spectra_preview()
                    except ValueError as e:
                        print(f"[ERROR] {self.fname} : {e}")
                        self.fig = None
            case SM4File.FileType.IZ:
                if len(self._spec.Current_y) > 0:
                    try:
                        self.spectra_preview()
                    except ValueError as e:
                        print(f"[ERROR] {self.fname} : {e}")
                        self.fig = None

        if self.fig is not None:
            save_dir = os.path.join(self.path.parent, 'sm4preview')
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            
            i = 0
            save_path = os.path.join(save_dir, self.fname)
            while os.path.isfile(f'{save_path}.png'):
                i += 1
                save_path = os.path.join(save_dir, f'{self.fname}_{i}')
            
            plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=150)
            plt.close()

    def image_preview(self, image: DataArray):
        self.fname = str(self.path.name).split('.')[0]
        self.fig = plt.figure(figsize=(11.3, 10))
        gs = self.fig.add_gridspec(nrows=100, ncols=7, hspace=0., wspace=0.)
        self.image_ax = self.fig.add_subplot(gs[0:65, 0:4])
        self.info_ax = self.fig.add_subplot(gs[65:100, 0:4])
        self.grid_ax = self.fig.add_subplot(gs[0:53, 4:7])
        self.spec_ax = self.fig.add_subplot(gs[53:100, 4:7])
        
        self.image_ax.axis('off')
        self.info_ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        self.info_ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        self.grid_ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        
        self.spec_ax.tick_params(left = False, labelleft = False, labelbottom = True, bottom = True)
        
        ## image axis
        image.spym.align()
        image.spym.plane()
        image.spym.fixzero()
        
        size = round(image.RHK_Xsize * abs(image.RHK_Xscale) * 1e9, 3)
        fontprops = fm.FontProperties(size=14)
        scalebar = AnchoredSizeBar(self.image_ax.transData,
                                   size/5, f'{size/5} nm', 'lower left',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=size/100,
                                   offset=1,
                                   fontproperties=fontprops)
        
        self.image_ax.imshow(image.data, extent=[0, size, 0, size], cmap='afmhot')
        self.image_ax.add_artist(scalebar)
        self.image_ax.set_ylabel(" ")
        
        ## Grid axis
        self.grid_ax.set(xlim=(-1, 1), ylim=(-1.057, 1.057))
        self.grid_ax.axhline(0, lw=.5, ls='-', c='gray')
        self.grid_ax.axvline(0, lw=.5, ls='-', c='gray')
        self.grid_ax.axvline(0.9775, lw=6.7, c='lightgray')
        self.grid_ax.axvline(-0.975, lw=6.7, c='lightgray')
        self.grid_ax.axhline(1.03, lw=6.7, c='lightgray')
        self.grid_ax.axhline(-1.03, lw=6.7, c='lightgray')

        match image.RHK_PiezoSensitivity_TubeCalibration:
            case '10K':
                calibration = 2220
            case '100K':
                calibration = 2220
            case 'Nitrogen':
                calibration = 2220
            case 'Helium':
                calibration = 2200
            case '300K':
                calibration = 12600
            case 'Room Temperature':
                calibration = 12600
            case _:
                print(f'Tube Calibration not accounted for: {image.RHK_PiezoSensitivity_TubeCalibration}')
                calibration = 2200

        offset = 1e9 * np.array([image.RHK_Xoffset, image.RHK_Yoffset])
        offset = offset - size/2
        offset = offset / (calibration / 2)
        self.grid_ax.add_patch(Rectangle(offset, size/(calibration / 2), size/ (calibration / 2), facecolor='none', edgecolor='red'))
        
        ## Spectroscopy axis
        self.spec_ax.set_facecolor('lightgray')
        self.spec_ax.yaxis.set_label_position("right")
        self.spec_ax.tick_params(axis="x", direction='in', top=True)
        match self.type:
            case SM4File.FileType.dIdV:
                self.spec_ax.set_xlabel('Voltage (mV)')
                self.spec_ax.set_ylabel('dI/dV (arb)')
            case SM4File.FileType.IZ:
                self.spec_ax.set_xlabel('Tip height (nm)')
                self.spec_ax.set_ylabel('Distance (nm)')

        ## Info panel
        self.info_ax.set_facecolor('gray')
        self.info_ax.text(0.01, 0.9, self.path.name, c='white')
        self.info_ax.text(0.01, 0.8, f'{image.RHK_Date} {image.RHK_Time}', c='white')
        self.info_ax.text(0.01, 0.7, f'Image width: {size} nm', c='white')
        self.info_ax.text(0.01, 0.6, f'Bias: {image.bias:.3f} V', c='white')
        self.info_ax.text(0.01, 0.5, f'Current: {image.setpoint} A', c='white')
        # self.info_ax.text(0.01, 0.4, f'Line time: {image.line}')

    def spectra_preview(self):
        if len(self.path.name.split("_")) < 7:
            return None
        
        src_dir = self.path.parent
        files = [x for x in os.listdir(src_dir) if x.endswith('.sm4')]
        dates = [x.split('.')[0].split('_') for x in files]
        dates = [x[-7:] for x in dates if len(x) > 7]
        dates = [datetime.datetime(*[int(d) for d in date]) for date in dates]
        dates = list(zip(dates, range(len(dates))))
        dates_sorted, permuted_indices = list(zip(*sorted(dates)))
        file_date = self.path.name.split('.')[0].split('_')[-7:]  # Date of the current file
        file_date = datetime.datetime(*[int(d) for d in file_date])
        
        files = [files[i] for i in list(permuted_indices)]
        idx = dates_sorted.index(file_date) # index of the current file in the date ordered list
        topography = None

        while idx >= 0:
            f = spym.load(os.path.join(src_dir, files[idx]))
            if f is None:
                idx -= 1
            elif 'data_vars' in f.__dir__():
                if 'Topography_Forward' in f.data_vars:
                    topography = f.Topography_Forward
                    if topography.data.shape[0] == topography.data.shape[1]: ### There is no full proof way to tell the difference between data that has only dIdV and data that has both image and dIdV - checking if the image is square is the closest option
                        line_average = np.average(topography.data, axis=1)
                        num_zeros = len(topography.data) - np.count_nonzero(line_average)
                        if num_zeros == 0:
                            break
                        else:
                            topography = None
                idx -= 1
            else:
                idx -= 1

        if topography is not None:
            self.image_preview(topography)
            match self.type:
                case SM4File.FileType.dIdV:
                    self.plot_spectra(self._spec, topography)
                case SM4File.FileType.IZ:
                    self.plot_iz(self._spec, topography)


    def plot_spectra(self, ldos: DataArray, topography: DataArray):
        if 'RHK_SpecDrift_Xcoord' not in ldos.attrs:
            return
        
        ## Spectra
        ldos_coords = self.unique_coordinates(zip(ldos.RHK_SpecDrift_Xcoord, ldos.RHK_SpecDrift_Ycoord))
        N = len(ldos_coords)
        if N == 0:
            return

        xsize = ldos.RHK_Xsize
        total = ldos.RHK_Ysize
        repetitions = total//N
        x = ldos.LIA_Current_x.data * 1e3
        ldos_ave = np.reshape(ldos.data, (xsize, N, repetitions))
        ldos_ave = np.mean(ldos_ave, axis=2).T

        ## Spec Coordinates
        xoffset = topography.RHK_Xoffset
        yoffset = topography.RHK_Yoffset
        xscale = topography.RHK_Xscale
        yscale = topography.RHK_Yscale
        xsize = topography.RHK_Xsize
        ysize = topography.RHK_Ysize
        width = np.abs(xscale * xsize)
        height = np.abs(yscale * ysize)

        offset = np.array([xoffset, yoffset]) + 0.5 * np.array([-width, -height])

        ## Plot
        skip = np.max(ldos_ave) / 10
        waterfall_offset = np.flip([i * skip for i in range(N)])
        colors = plt.cm.jet(np.linspace(0, 1, N))
        
        self.spec_ax.set_facecolor('white')
        for (i, (dIdV, real_coord)) in enumerate(zip(ldos_ave, ldos_coords)):
            view_coord = np.array(real_coord - offset) * 1e9

            self.spec_ax.plot(x, dIdV + waterfall_offset[i], c=colors[i])
            self.image_ax.plot(view_coord[0], view_coord[1], marker="o", c=colors[i])

    def plot_iz(self, iz: DataArray, topography: DataArray):
        if self.type is not SM4File.FileType.IZ:
            print("File contains no IZ data.")
            return 
        
        if 'RHK_SpecDrift_Xcoord' not in iz.attrs:
            print('RHK_SpecDrift_Xcoord not in Current attributes.')
            return

        coords = self.unique_coordinates(zip(iz.RHK_SpecDrift_Xcoord, iz.RHK_SpecDrift_Ycoord))
        N = len(coords)
        if N == 0:
            print("No IZ data found.")
            return

        xsize = iz.RHK_Xsize
        total = iz.RHK_Ysize
        repetitions = total//N
        x = iz.Current_x.data * 1e9
        iz_data = iz.data.reshape(xsize, N, repetitions).T
        approach = iz_data[::2, :].mean(axis=0)
        retract = iz_data[1::2, :].mean(axis=0)
        iz_ave = np.flip(retract - approach)

        ## Spec Coordinates
        xoffset = topography.RHK_Xoffset
        yoffset = topography.RHK_Yoffset
        xscale = topography.RHK_Xscale
        yscale = topography.RHK_Yscale
        xsize = topography.RHK_Xsize
        ysize = topography.RHK_Ysize
        width = np.abs(xscale * xsize)
        height = np.abs(yscale * ysize)

        offset = np.array([xoffset, yoffset]) + 0.5 * np.array([-width, -height])
        line_cut = (coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
        line_length = np.sqrt(line_cut[0]**2 + line_cut[1]**2) * 1e9

        ## Plot
        if len(coords) > 1:
            aspect = abs(x[-1] - x[0]) / line_length
            self.spec_ax.yaxis.set_ticks_position('right')
            self.spec_ax.tick_params(axis="y", direction='in')
            self.spec_ax.imshow(iz_ave, cmap='seismic', aspect=aspect, extent=[x[0], x[-1], 0, line_length], norm=CenteredNorm())
            
            (x1, y1) = np.array(coords[0] - offset) * 1e9
            (x2, y2) = np.array(coords[-1] - offset) * 1e9
            self.image_ax.arrow(x1, y1, x2 - x1, y2 - y1, lw=0.1, width=0.2, length_includes_head=True, edgecolor='w', facecolor='w')
        else:
            self.spec_ax.set_facecolor('white')
            self.spec_ax.plot(x, approach[0])
            self.spec_ax.plot(x, retract[0])

            view_coord = np.array(coords[0] - offset) * 1e9
            self.image_ax.plot(view_coord[0], view_coord[1], marker="o")

    def unique_coordinates(self, coords):
        seen = set()
        seen_add = seen.add
        return [x for x in coords if not (x in seen or seen_add(x))]
