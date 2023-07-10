import spym
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from pathlib import Path
from dataclasses import dataclass
from xarray import DataArray

@dataclass
class RawImage:
    forward: DataArray
    reverse: DataArray

class SM4File:
    def __init__(self, src_path: str):
        self.path = Path(src_path)
        assert [self.path.suffix == '.sm4', "Must provide a .sm4 file to SM4File class."]
        
        file = spym.load(src_path)
        self.topography = RawImage(forward=file.Topography_Forward, reverse=file.Topography_Backward)
        self.current = RawImage(forward=file.Current_Forward, reverse=file.Current_Backward)
        self.lia = RawImage(forward=file.LIA_Current_Forward, reverse=file.LIA_Current_Backward)
        
    def generate_preview(self):
        fig = plt.figure(figsize=(11.6, 10))
        gs = fig.add_gridspec(6, 7, hspace=0., wspace=0.)
        topo_ax = fig.add_subplot(gs[0:4, 0:4])
        info_ax = fig.add_subplot(gs[4:6, 0:4])
        grid_ax = fig.add_subplot(gs[0:3, 4:7])
        spec_ax = fig.add_subplot(gs[3:6, 4:7])
        
        topo_ax.axis('off')
        info_ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        info_ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        grid_ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        spec_ax.yaxis.tick_right()
        
        self.topography.forward.spym.align()
        self.topography.forward.spym.plane()
        self.topography.forward.spym.fixzero()
        
        size = round(self.topography.forward.RHK_Xsize * abs(self.topography.forward.RHK_Xscale) * 1e9, 3)
        fontprops = fm.FontProperties(size=14)
        scalebar = AnchoredSizeBar(topo_ax.transData,
                                   size/5, f'{size/5} nm', 'lower left',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=size/100,
                                   offset=1,
                                   fontproperties=fontprops)
        
        topo_ax.imshow(self.topography.forward.data, extent=[0, size, 0, size], cmap='afmhot')
        topo_ax.add_artist(scalebar)
        topo_ax.set_ylabel(" ")
        
        # Set identical scales for both axes
        grid_ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal')
        grid_ax.axhline(0, lw=.5, ls='-', c='gray')
        grid_ax.axvline(0, lw=.5, ls='-', c='gray')
        
        offset = 1e9 * np.array([self.topography.forward.RHK_Xoffset, self.topography.forward.RHK_Yoffset])
        offset = offset - size
        offset = offset / 1110
        grid_ax.add_patch(Rectangle(offset, size/1110, size/1110, facecolor='none', edgecolor='red'))
        
        ##
        info_ax.set_facecolor('gray')
        info_ax.text(0.01, 0.9, self.path.name, c='white')
        info_ax.text(0.01, 0.8, f'{self.topography.forward.RHK_Date} {self.topography.forward.RHK_Time}', c='white')
        info_ax.text(0.01, 0.7, f'Image width: {size} nm', c='white')
        info_ax.text(0.01, 0.6, f'Bias: {self.topography.forward.bias:.3f} V', c='white')
        info_ax.text(0.01, 0.5, f'Current: {self.topography.forward.setpoint} A', c='white')
        
        plt.savefig(f'{self.path.name}.png', bbox_inches='tight', dpi=300)