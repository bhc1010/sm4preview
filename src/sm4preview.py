import os, datetime
import spym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from pathlib import Path
from dataclasses import dataclass
from xarray import DataArray
from enum import Enum

@dataclass
class RawImage:
    forward: DataArray
    reverse: DataArray

@dataclass
class RawSpec:
    current: DataArray
    lia: DataArray

class SM4File:
    FileType = Enum("FileType", ["Image", "Spectra"])
    def __init__(self, src_path: str):
        self.path = Path(src_path)
        self.file = spym.load(src_path)
        self.fig = None
        self.type = None

        try:
            data_vars = self.file.data_vars
        except:
            print(f"Invalid file, no data_vars from {src_path}")
            return

        if "Current" not in data_vars:
            self.type = SM4File.FileType.Image
        else:
            self.type = SM4File.FileType.Spectra

        self.parse_data(self.type)

    def parse_data(self, T: FileType):
        match T:
            case SM4File.FileType.Image:
                self._topography = RawImage(forward=self.file.Topography_Forward, reverse=self.file.Topography_Backward)
                # self._lia_map = RawImage(forward=self.file.LIA_Current_Forward, reverse=self.file.LIA_Current_Backward)
            case SM4File.FileType.Spectra:
                self._spec = RawSpec(current=self.file.Current, lia=self.file.LIA_Current)
            case _:
                pass
        
    def generate_preview(self):
        match self.type:
            case SM4File.FileType.Image:
                self.image_preview(self._topography.forward)
            case SM4File.FileType.Spectra:
                self.spectra_preview()

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

        # if self.type == SM4File.FileType.Image:
        #     self.image_preview(self._lia_map.forward)
        #     plt.savefig(f'{self.path.name}_ldos_map.png', bbox_inches='tight', dpi=150)

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
        
        self.spec_ax.tick_params(left = False, right = False , labelleft = False, labelbottom = True, bottom = True)
        
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
        
        match self._topography.forward.RHK_PiezoSensitivity_TubeCalibration:
            case '10K':
                calibration = 2220
            case '300K':
                calibration = 12600

        offset = 1e9 * np.array([image.RHK_Xoffset, image.RHK_Yoffset])
        offset = offset - size/2
        offset = offset / (calibration / 2)
        self.grid_ax.add_patch(Rectangle(offset, size/(calibration / 2), size/ (calibration / 2), facecolor='none', edgecolor='red'))
        
        ## Spectroscopy axis
        self.spec_ax.set_facecolor('lightgray')
        self.spec_ax.yaxis.set_label_position("right")
        self.spec_ax.tick_params(axis="x", direction='in', top=True)
        self.spec_ax.set_xlabel('Voltage (mV)')
        self.spec_ax.set_ylabel('dI/dV (arb)')

        ## Info panel
        self.info_ax.set_facecolor('gray')
        self.info_ax.text(0.01, 0.9, self.path.name, c='white')
        self.info_ax.text(0.01, 0.8, f'{image.RHK_Date} {image.RHK_Time}', c='white')
        self.info_ax.text(0.01, 0.7, f'Image width: {size} nm', c='white')
        self.info_ax.text(0.01, 0.6, f'Bias: {image.bias:.3f} V', c='white')
        self.info_ax.text(0.01, 0.5, f'Current: {image.setpoint} A', c='white')

    def spectra_preview(self):
        src_dir = self.path.parent
        files = [x for x in os.listdir(src_dir) if x.endswith('.sm4')]
        current_year = str(datetime.date.today().year)
        dates = [current_year + x.split(current_year)[-1].split('.')[0] for x in files]
        dates = [datetime.datetime(*[int(d) for d in x.split("_")]) for x in dates]
        dates = list(zip(dates, range(len(dates))))
        dates_sorted, permuted_indices = list(zip(*sorted(dates)))
        file_date = current_year + self.path.name.split(current_year)[-1].split('.')[0]
        file_date = datetime.datetime(*[int(d) for d in file_date.split("_")])
        
        files = [files[i] for i in list(permuted_indices)]
        idx = dates_sorted.index(file_date)
        topography = None

        while idx > 0:
            f = spym.load(os.path.join(src_dir, files[idx-1]))
            if 'Topography_Forward' in f.data_vars:
                if 'Current' not in f.data_vars:
                    topography = f.Topography_Forward
                    line_average = np.average(topography.data, axis=1)
                    num_zeros = len(topography.data) - np.count_nonzero(line_average)
                    if num_zeros == 0:
                        break
                    else:
                        topography = None
            idx -= 1

        if topography is not None:
            self.image_preview(topography)
            self.fname = str(files[idx-1]).split('.')[0] + '_spec'

            ldos = self._spec.lia
            self.plot_spectra(ldos, topography)

    def plot_spectra(self, ldos: DataArray, topography: DataArray):
        ldos_coords = self.unique_coordinates(zip(ldos.RHK_SpecDrift_Xcoord, ldos.RHK_SpecDrift_Ycoord))
        xsize = ldos.RHK_Xsize
        total = ldos.RHK_Ysize
        N = len(ldos_coords)
        repetitions = total//N

        x = ldos.LIA_Current_x.data * 1e3
        ldos_ave = np.array([np.zeros(xsize)]*N)
        for i in range(N):
            for j in range(repetitions):
                ldos_ave[i] += ldos.data.transpose()[i*repetitions+j]
            ldos_ave[i] /= repetitions

        skip = np.max(ldos_ave) / 10
        offset = np.flip([i * skip for i in range(N)])
        # offset = np.zeros(N)

        colors = plt.cm.jet(np.linspace(0, 1, N))
        
        for (i,dIdV) in enumerate(ldos_ave):
            self.spec_ax.plot(x, dIdV + offset[i], c=colors[i])

        self.spec_ax.set_facecolor('white')

        ## Plot points
        xoffset = topography.RHK_Xoffset
        yoffset = topography.RHK_Yoffset
        offset = np.array([xoffset, yoffset])
        xscale = topography.RHK_Xscale
        yscale = topography.RHK_Yscale
        xsize = topography.RHK_Xsize
        ysize = topography.RHK_Ysize
        width = np.abs(xscale * xsize)
        height = np.abs(yscale * ysize)
        offset += 0.5 * np.array([-width, -height])

        for (i, real_coord) in enumerate(ldos_coords):
            view_coord = np.array(real_coord - offset) * 1e9
            self.image_ax.plot(view_coord[0], view_coord[1], marker="o", c=colors[i])

    def unique_coordinates(self, coords):
        seen = set()
        seen_add = seen.add
        return [x for x in coords if not (x in seen or seen_add(x))]