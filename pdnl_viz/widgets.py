
import os
import time
import tempfile
import random

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure

from PyQt6 import QtCore, QtWidgets

import matplotlib as mpl
from matplotlib import pyplot as plt

from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
import matplotlib.collections as mcollections

import numpy as np
from tqdm import tqdm

import pdnl_sana as sana
import pdnl_sana.logging
import pdnl_sana.slide
import pdnl_sana.image
import pdnl_sana.geo
import pdnl_sana.process
import pdnl_sana.filter
import pdnl_sana.segment
import pdnl_sana.quantify
import pdnl_sana.interpolate

import neuseg.tissue
import neuseg.nuclei


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, tmp_directory=None, **kwargs):
        self.figure = Figure(figsize=(width, height), **kwargs)
        super().__init__(self.figure)
        if tmp_directory is None:
            self.temp_directory = tempfile.TemporaryDirectory()
            self.tmp_directory = self.temp_directory.name
            self.tmp_directory = './test_tmp'
        else:
            self.tmp_directory = tmp_directory
        self.level = 0
        self.n_cores = 1
        self.frame_size = 1024

    def start_tracking(self):
        self.cidpress = self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
    def stop_tracking(self):
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)
    def on_press(self, event):
        pass
    def on_release(self, event):
        pass
    def on_motion(self, event):
        pass
    def set_slide(self, loader):
        self.loader = loader
        self.tb = self.loader.load_thumbnail()
        self.input_slide = self.loader.fname
        self.logger = sana.logging.Logger('normal', os.path.join(self.tmp_directory, 'log.pkl'))

    def update_plot(self):
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def find_global_threshold(self, input_slide: str, target_stain: str, **kwargs):
        results, nresults = sana.process.preprocess_wsi(
            input_slide=input_slide, staining_code=self.staining_code, 
            target_stain=target_stain, ret_indices=True, shuffle=True, **kwargs)
        
        done = 0
        last_pct = 0
        histograms = []
        done_mask = sana.image.frame_like(self.tb, np.zeros_like(self.tb.img[:,:,0]))
        frame_size_tb = int(self.frame_size / self.loader.ds[self.tb.level])
        for result in results:
            done += 1
            if not result is None:
                hist, i, j = result
                histograms.append(hist)
                done_mask.img[
                    j*frame_size_tb:j*frame_size_tb+frame_size_tb, 
                    i*frame_size_tb:i*frame_size_tb+frame_size_tb] = 1
                done_mask.mask(self.tissue_mask)
            pct = (100*done) // nresults

            # TODO: mask the tissue mask by processed_mask and plot these polygons
            if pct > last_pct or done == nresults+1:
                last_pct = pct
                done_polys = done_mask.to_polygons()[0]
                self.ax.clear()
                self.ax.imshow(self.tb.img)
                [self.ax.plot(*x.T, color='black') for x in done_polys]
                self.ax.set_title(f'Preprocessing Chunks... {last_pct}%')
                self.update_plot()

        self.ax.set_title('Done Preprocessing.')
        p = self.tissue_mask.to_polygons()[0]
        self.ax.clear()
        self.ax.imshow(self.tb.img)
        [self.ax.plot(*x.T, color='black') for x in p]
        self.update_plot()

        # calculate the stain threshold for the image
        global_threshold = sana.threshold.triangular_method(
            np.mean(histograms, axis=0)[:,0], 
            strictness=self.loader.logger.data.get('strictness', -0.8))
        self.logger.debug(f"Global Stain Threshold: {global_threshold}")
        self.logger.data['global_threshold'] = global_threshold
        self.logger.write_data()

        return global_threshold

    def preprocess_wsi(self, target_stain):
        self.tissue_mask = neuseg.tissue.get_tissue_mask(self.tb)[0]
        self.tissue_mask.save(os.path.join(self.tmp_directory, 'tissue_mask.png'))
        self.rois, self.roi_holes = self.tissue_mask.to_polygons()
        self.rois = {'Tissue': self.rois}
        # for x in self.rois['Tissue']:
        #    self.ax.plot(*x.T, color='black')
        # self.update_plot()

        # get the coordinates in the WSI to load chunks from
        global_threshold = self.find_global_threshold(
            self.input_slide, target_stain=target_stain, frame_size=self.frame_size, level=self.level,
            rois=self.rois, roi_holes=self.roi_holes,
            tmp_directory=self.tmp_directory, n_cores=self.n_cores) 

        return global_threshold       

class StainWidget(MplCanvas):
    def __init__(self, parent, width=5, height=4, **kwargs):
        super().__init__(parent=parent, width=width, height=height, **kwargs)
        self.ax_frame = self.figure.add_subplot(221)
        self.ax_recon = self.figure.add_subplot(222)
        self.ax_hist = self.figure.add_subplot(223)
        self.ax_overlay = self.figure.add_subplot(224)

        self.strictness = 0.0
        self.apply_smoothing = True
        self.normalize_background = True
        self.background_radius = 100
        self.background_overlap = 0.5

    def set_slide(self, loader, loc, size):
        super().set_slide(loader)
        self.frame = self.loader.load_frame(loc, size)
        self.ax_frame.imshow(self.frame.img)
        self.set_processor()

    def set_processor(self):
        self.processor = pdnl_sana.process.HDABProcessor(
            self.loader.logger, self.frame, 
            subtract_dab=True,
            apply_smoothing=self.apply_smoothing, 
            normalize_background=self.normalize_background, 
            radius=self.background_radius, 
            overlap=self.background_overlap, 
        )
        ss = self.processor.ss
        hem = self.processor.hem.img
        dab = self.processor.dab.img
        hem_od = hem.astype(float) * (ss.max_od[0] - ss.min_od[0]) / 255 + ss.min_od[0]        
        dab_od = dab.astype(float) * (ss.max_od[1] - ss.min_od[1]) / 255 + ss.min_od[1]
        res_od = np.zeros_like(dab_od)
        self.stains = np.concatenate([hem_od, dab_od, res_od], axis=2)

    def set_threshold(self, target_stain):
        if hasattr(self, 'closing_radius'):
            filters = [
                pdnl_sana.filter.MorphologyFilter('closing', 'ellipse', self.closing_radius),
                pdnl_sana.filter.MorphologyFilter('opening', 'ellipse', self.opening_radius),
            ]
        else:
            filters = []

        hist = self.stain.get_histogram(mask=self.processor.main_mask)
        self.ax_hist.clear()
        _ = pdnl_sana.threshold.triangular_method(hist, strictness=self.strictness, ax=self.ax_hist)
        ret = self.processor.run(triangular_strictness=self.strictness, morphology_filters=filters, target_stain=target_stain)
        pos = ret['positive_stain']
        overlay = self.frame.copy(); overlay.blend(pos, color=(255,0,0), alpha=0.8)
        self.ax_overlay.imshow(overlay.img)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

class CSWidget(StainWidget):
    def __init__(self, parent, width=5, height=4, **kwargs):
        super().__init__(parent=parent, width=width, height=height, **kwargs)

        self.closing_radius = 0
        self.opening_radius = 0

    def set_processor(self):
        super().set_processor()
        self.stain = self.processor.hem
        self.stains[:,:,1] = 0
        self.stains[:,:,2] = 0
        recon = self.processor.ss.combine(self.stains)
        self.ax_recon.imshow(recon)
        self.set_threshold()
    def set_threshold(self):
        super().set_threshold('HEM')

class DABWidget(StainWidget):
    def __init__(self, parent, width=5, height=4, **kwargs):
        super().__init__(parent=parent, width=width, height=height, **kwargs)

    def set_processor(self):
        super().set_processor()
        self.stain = self.processor.dab
        self.stains[:,:,0] = 0
        self.stains[:,:,2] = 0
        recon = self.processor.ss.combine(self.stains)
        self.ax_recon.imshow(recon)
        self.set_threshold()
    def set_threshold(self):
        super().set_threshold('DAB')

class OverlayWidget(MplCanvas):
    def __init__(self, parent, width=5, height=4, **kwargs):
        super().__init__(parent=parent, width=width, height=height, **kwargs)
        self.ax = self.figure.add_subplot(111)
        self.ds_thumbnail = 2.0        
        self.window_size = 50
        self.alpha = 1.0

    def set_slide(self, loader, staining_code):
        super().set_slide(loader)
        self.staining_code = staining_code
        self.dab_wsi = None
        self.global_threshold = None
        self.dab_mask = None
        self.dab_tb = None
        self.ax.imshow(self.tb.img)
        self.update_plot()

    def process_dab(self, reprocess=True):
        if 'dab_global_threshold' in self.logger.data and not reprocess:
            self.global_threshold = self.logger.data['dab_global_threshold']
        else:
            self.global_threshold = super().preprocess_wsi('DAB')
            self.logger.data['dab_global_threshold'] = self.global_threshold
            self.logger.write_data()

        dab_mask_fpath = os.path.join(self.tmp_directory, 'dab_mask.npy')
        if os.path.exists(dab_mask_fpath) and not reprocess:
            self.dab_mask = sana.image.Frame(dab_mask_fpath)
        else:
            dab_frames = [os.path.join(self.tmp_directory, x) \
                          for x in os.listdir(self.tmp_directory) \
                            if x.startswith('dab_') and not 'mask' in x]
            w, h = self.loader.level_dimensions[0]
            self.dab_mask = sana.image.Frame(np.zeros((int(h//self.ds_thumbnail), int(w//self.ds_thumbnail), 1), dtype=bool))
            frame_size = int(self.frame_size//self.ds_thumbnail)
            for count, f in tqdm(enumerate(dab_frames), desc='processing chunks'): 
                i,j = list(map(int, f.replace('.png', '').split('_')[-2:]))

                stain = pdnl_sana.image.Frame(f)
                stain.resize(stain.size()//self.ds_thumbnail)
                stain.threshold(self.global_threshold)
                mask = pdnl_sana.image.Frame(f.replace('dab_', 'mask_').replace('.png', '.npz'))
                #stain.mask(mask)                
                
                y0 = j*frame_size
                y1 = np.clip(y0+frame_size, None, int(h//self.ds_thumbnail))
                x0 = i*frame_size
                x1 = np.clip(x0+frame_size, None, int(w//self.ds_thumbnail))

                self.dab_mask.img[y0:y1, x0:x1, :] = \
                    stain.img[0:y1-y0, 0:x1-x0, :] != 0
                pct = int(100*count / len(dab_frames))
                self.ax.set_title(f'Thresholding Chunks... {pct}%')
                self.update_plot()
                
            self.dab_mask.save(dab_mask_fpath)
            self.ax.set_title(f'Done Threshold.')
            self.update_plot()
        self.dab_mask.img = (self.dab_mask.img != 0).astype(bool)

    def aggregate_dab(self, reprocess=True):
        if self.dab_mask is None:
            self.process_dab(reprocess=reprocess)

        window_size = pdnl_sana.geo.Point(self.window_size, self.window_size, is_micron=True, level=0)
        window_size = self.loader.converter.to_pixels(window_size, level=0).astype(int)
        window_size_mask = (window_size // self.ds_thumbnail).astype(int)
        window = pdnl_sana.image.frame_like(self.tb, np.zeros((window_size_mask[1], window_size_mask[0], 1), dtype=float))
        window_mask = pdnl_sana.image.frame_like(self.tb, np.zeros((window_size_mask[1], window_size_mask[0], 1), dtype=float))
        window.level = 0
        k = window_size[0]
        if k % 2 == 0:
            k += 1
        kernel = pdnl_sana.filter.get_gaussian_kernel(k, window_size_mask[0]//2)
        tile_step = self.dab_mask.size() / self.tb.size() * self.ds_thumbnail

        self.dab_tb = self.dab_mask.convolve(kernel, tile_step, iterate=True, align_center=True, normalize=False)

        self.dab_tb = sana.image.frame_like(self.tb, self.dab_tb)
        self.dab_tb.resize(self.tb.size())

    def plot_dab(self, reprocess=True):
        if self.dab_tb is None:
            self.aggregate_dab(reprocess=reprocess)

        colormap = plt.cm.inferno
        olay = colormap(self.dab_tb.img[:,:,0] / np.max(self.dab_tb.img))[:,:,:3]
        overlay = self.tb.copy()
        overlay.img = self.alpha*(255*olay) + (1-self.alpha)*self.tb.img
        overlay.img = overlay.img.astype(np.uint8)

        self.ax.imshow(overlay.img)
        self.update_plot()

class ROIWidget(MplCanvas):
    def __init__(self, parent, anno_file, width=5, height=4, **kwargs):
        super().__init__(parent=parent, width=width, height=height, **kwargs)
        self.ax_frame = self.figure.add_subplot(221)
        self.ax_curve = self.figure.add_subplot(222)
        self.ax_dab = self.figure.add_subplot(223)
        self.ax_hem = self.figure.add_subplot(224)

        annotations = sana.utils.read_geojson(anno_file)
        self.csf = [x for x in annotations if x.class_name == 'CSF'][0].to_curve()
        self.r = [x for x in annotations if x.class_name == 'R'][0].to_curve()
        self.wm = [x for x in annotations if x.class_name == 'WM'][0].to_curve()
        self.l = [x for x in annotations if x.class_name == 'L'][0].to_curve()
        self.segs = [self.csf, self.r, self.wm, self.l]

        self.strictness = 0.0
        self.apply_smoothing = True
        self.normalize_background = True
        self.background_radius = 100
        self.background_overlap = 0.5
        self.minimum_soma_radius = 3        
        self.maximum_soma_radius = 20
        self.maximum_soma_area = np.pi*self.maximum_soma_radius**2
        self.cs_strictness = -0.8
        self.cs_closing_radius = 2
        self.cs_opening_radius = 2
        self.nlayers = 50

    def set_slide(self, loader):
        super().set_slide(loader)
        for x in self.segs:
            x.level = 2
            x.is_micron = False
            self.loader.converter.rescale(x, 0)

        # TODO: plot laynii-like layers
        self.frame = self.loader.load_frame_with_segmentations(*self.segs)
        [sana.geo.transform_array_with_logger(x, self.loader.logger) for x in self.segs]
        self.ax_frame.imshow(self.frame.img)
        [self.ax_frame.plot(*x.T) for x in self.segs]
        self.update_plot()

    def set_deform(self):
        self.ax_curve.set_title('Calculating Layers...')
        self.update_plot()
        self.sample_grid, _ = sana.interpolate.fan_sample(*self.segs)
        self.set_layer_masks()

    def set_layer_masks(self):
        out_h = self.frame.img.shape[0]
        out_w = self.frame.img.shape[1]
        self.layer_masks = sana.interpolate.sample_grid_to_layers(
            self.sample_grid, out_h=out_h, out_w=out_w, nlayers=self.nlayers)

        self.ax_frame.clear()
        self.ax_frame.imshow(self.layer_masks, cmap='rainbow')
        self.ax_curve.set_title('')
        self.update_plot()

    def process_stains(self):
        self.process_dab()
        self.plot_dab_curves()
        self.process_cs()
        self.plot_cs_curves()

    def process_dab(self):
        self.processor = pdnl_sana.process.HDABProcessor(
            self.loader.logger, self.frame, 
            apply_smoothing=self.apply_smoothing,
            normalize_background=self.normalize_background, 
            radius=self.background_radius, 
            overlap=self.background_overlap, 
        )
        if hasattr(self, 'closing_radius'):
            filters = [
                pdnl_sana.filter.MorphologyFilter('closing', 'ellipse', self.closing_radius),
                pdnl_sana.filter.MorphologyFilter('opening', 'ellipse', self.opening_radius),
            ]
        else:
            filters = []
        self.stain = self.processor.dab
        hist = self.stain.get_histogram(mask=self.processor.main_mask)
        _ = pdnl_sana.threshold.triangular_method(hist, strictness=self.strictness)
        ret = self.processor.run(triangular_strictness=self.strictness, morphology_filters=filters, target_stain="DAB")
        pos = ret['positive_stain']
        self.pos_dab = pos
        self.ax_dab.imshow(self.pos_dab.img)
        self.update_plot()

    def process_cs(self):
        self.processor = pdnl_sana.process.HDABProcessor(
            self.loader.logger, self.frame, 
            apply_smoothing=self.apply_smoothing,
            normalize_background=self.normalize_background, 
            radius=self.background_radius, 
            overlap=self.background_overlap, 
        )
        if hasattr(self, 'cs_closing_radius'):
            filters = [
                pdnl_sana.filter.MorphologyFilter('closing', 'ellipse', self.cs_closing_radius),
                pdnl_sana.filter.MorphologyFilter('opening', 'ellipse', self.cs_opening_radius),
            ]
        else:
            filters = []
        self.stain = self.processor.hem
        hist = self.stain.get_histogram(mask=self.processor.main_mask)
        ret = self.processor.run(triangular_strictness=self.cs_strictness, morphology_filters=filters, target_stain="HEM")
        pos = ret['positive_stain']
        self.pos_hem = pos
    
        polys = self.pos_hem.to_polygons()[0]
        polys = [x for x in polys if x.get_area() <= self.maximum_soma_area]
        self.pos_hem = sana.image.create_mask_like(self.pos_hem, polys)

        ctrs = sana.segment.detect_somas(self.pos_hem, minimum_soma_radius=self.minimum_soma_radius)
        polys, _ = self.pos_hem.instance_segment(ctrs)[0]

        bbs = [p.bounding_box() for p in polys]
        ctrs = np.array([loc + size // 2 for (loc, size) in bbs])
        areas = np.array([p.get_area() for p in polys])
        ints = []
        for p in polys:
            loc, size = p.bounding_box()
            tile = sana.image.Frame(self.stain.get_tile(loc, size))
            p.translate(loc)
            tile_mask = sana.image.create_mask_like(tile, [p])
            p.translate(-loc)
            ints.append(np.mean(tile.img[tile_mask.img != 0]))
        ints = np.array(ints)

        self.cs_cells = np.vstack([ctrs[:,0], ctrs[:,1], areas, ints]).T
        self.ax_hem.imshow(self.pos_hem.img)
        self.update_plot()

    def plot_dab_curves(self):
        k = int(self.nlayers / 7)
        if k % 2 == 0: k += 1
        def smooth(x, N):
            mu, sg = np.nanmean(x), np.nanstd(x)
            x = (x-mu) / sg
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)
        dab_curve = sana.quantify.apply_layer_masks_to_frame(self.pos_dab, self.layer_masks)
        self.ax_curve.clear()
        self.ax_curve.plot(smooth(dab_curve, k), color='blue', label='DAB')
        self.update_plot()

    def plot_cs_curves(self):
        k = int(self.nlayers / 7)
        if k % 2 == 0: k += 1
        def smooth(x, N):
            mu, sg = np.nanmean(x), np.nanstd(x)
            x = (x-mu) / sg
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        labels = ['CS Cell Density', 'CS Cell Area', 'CS Cell Intensity']
        colors = ['red', 'green', 'yellow']
        cell_features = sana.quantify.apply_layer_masks_to_cells(self.cs_cells, self.layer_masks)
        for i in range(3):
            self.ax_curve.plot(smooth(cell_features[:,i], k), color=colors[i], label=labels[i])
        self.ax_curve.legend()
        self.update_plot()

class ThumbnailWidget(MplCanvas):
    updated = QtCore.Signal()
    zoom_reset = True
    t0 = time.time()
    fps = 10
    def __init__(self, parent, width=5, height=4, **kwargs):
        super().__init__(parent=parent, width=width, height=height, **kwargs)
        self.ax = self.figure.add_subplot(111)
        self.rect = Rectangle(xy=(0,0), width=1, height=1, color='red', linewidth=1, linestyle='--', fill=False)
        self.rect.moving = False
        self.ax.add_patch(self.rect)

    def set_slide(self, loader):
        self.loader = loader 

        self.tb = self.loader.load_thumbnail()
        self.ax.imshow(self.tb.img)        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        self.ctr = self.tb.size() // 2
        l = 400
        self.rect_size = pdnl_sana.geo.Point(l, l, is_micron=False, level=0)
        self.rect_size_tb = self.loader.converter.rescale(self.rect_size, self.loader.thumbnail_level)
        self.rect.set_xy(self.ctr)
        self.rect.set_width(self.rect_size_tb[0])
        self.rect.set_height(self.rect_size_tb[1])

        self.start_tracking()        
        self.updated.emit()

    def get_rect(self):
        loc = pdnl_sana.geo.Point(*self.rect.get_xy(), is_micron=False, level=self.loader.thumbnail_level)
        return loc, self.rect_size
        
    def on_press(self, event):
        if self.rect is None or (event.inaxes != self.rect.axes): return

        if event.button is MouseButton.RIGHT: 
            l = 250
            self.loader.converter.mtop(l, level=self.tb.level)
            x, y = event.xdata, event.ydata
            self.ax.set_xlim([x-l//2, x+l//2])
            self.ax.set_ylim([y+l//2, y-l//2])
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
        else:
            self.rect.set_xy((event.xdata-self.rect_size_tb[0]//2, event.ydata-self.rect_size_tb[1]//2))
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.updated.emit()

    def on_motion(self, event):
        if self.rect is None or (event.inaxes != self.rect.axes and not self.zoom_reset): 
            self.ax.set_xlim([0, self.tb.size()[0]])
            self.ax.set_ylim([self.tb.size()[1], 1])
            self.zoom_reset = True
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            return
        
        self.zoom_reset = False
        dt = time.time() - self.t0
        if dt < 1/self.fps: return

        self.t0 = time.time()

    def on_release(self, event):
        # self.rect.moving = False
        # self.figure.canvas.draw()
        # self.figure.canvas.flush_events()
        pass

class NeusegWidget(MplCanvas):
    def __init__(self, parent, width=5, height=4, **kwargs):
        super().__init__(parent=parent, width=width, height=height, **kwargs)
        self.ax = self.figure.add_subplot(111)

        self.reset_data()

    def reset_data(self):
        self.tissue_mask = None
        self.rois, self.roi_holes = None, None
        self.mi_soma_r = 2.0
        self.mx_soma_r = 10.0
        self.close_r = 2
        self.open_r = 2
        self.feature_ds = 4.0
        self.window_size = 1000
        self.heatmap_mi = np.full((3,), -1.0)
        self.heatmap_mx = np.full((3,), +2.5)
        self.heatmap_alpha = 1.0
        self.segment_length = 1000
        self.loc_x = None
        self.loc_y = None

    def set_slide(self, loader, staining_code):
        super().set_slide(loader)
        self.staining_code = staining_code        
        self.ax.imshow(self.tb.img)
        self.update_plot()

    def preprocess_cs(self):
        super().preprocess_wsi('HEM')

    def segment_cells(self):
        if not 'global_threshold' in self.logger.data:
            self.preprocess_cs()

        frame_idxs = [list(map(int, os.path.splitext(x)[0].split('_')[1:3])) \
                      for x in os.listdir(self.tmp_directory) if 'counterstain' in x]
        random.shuffle(frame_idxs)
        job_args = [{
            'threshold': self.logger.data['global_threshold'],
            'closing_radius': self.close_r,
            'opening_radius': self.open_r,
            'minimum_soma_radius': self.mi_soma_r,
            'maximum_soma_radius': self.mx_soma_r,
            'tmp_directory': self.tmp_directory, 
            'j': j, 'i': i,
        } for (j,i) in frame_idxs]

        cells = []
        done = 0
        last_pct = 0
        plot_cells = []
        for res in sana.utils.dispatch_jobs(neuseg.segment_nuclei_chunk, job_args, n_cores=self.n_cores, progress_str="Segment Cells"):
            done += 1
            pct = (100*done) // len(job_args)

            plot_cells.append(res)        
            cells.append(res)
            if pct > last_pct:
                for x in plot_cells:
                    self.ax.plot(*x[::16].T / self.loader.ds[self.tb.level], 
                                 linestyle='', marker='.', color='red')
                    self.update_plot()
                    plot_cells = []
        self.cells = np.concatenate(cells, axis=0)
        np.save(os.path.join(self.tmp_directory, 'cells.npy'), self.cells)

    def aggregate_cells(self):
        cells_fpath = os.path.join(self.tmp_directory, 'cells.npy')
        if not os.path.exists(cells_fpath):
            self.segment_cells()
        self.cells = np.load(cells_fpath)
        self.tissue_mask = sana.image.Frame(os.path.join(self.tmp_directory, 'tissue_mask.png'))

        self.heatmap = neuseg.nuclei.aggregate_nuclei_features(
            cells=self.cells, tb=self.tb, 
            mpp=self.loader.converter.mpp,
            ds=self.loader.converter.ds,
            level_dimensions=self.loader.level_dimensions,
            logger=self.logger,
            ds_thumbnail=self.feature_ds,
            window_size=self.window_size,
            n_cores=self.n_cores,
        )
        self.heatmap.level = self.tb.level
        self.heatmap.converter = self.tb.converter
        self.heatmap.resize(self.tb.size())
        self.heatmap.save(os.path.join(self.tmp_directory, 'heatmap.npy'))
        self.heatmap.mask(self.tissue_mask)
        self.plot_heatmap()

    def plot_heatmap(self):
        self.ax.clear()
        x = self.heatmap.copy().img
        valid = x[self.tissue_mask.img[:,:,0] != 0]
        mu = np.nanmean(valid, axis=0)
        sg = np.nanstd(valid, axis=0)
        mi = (mu+self.heatmap_mi*sg)
        mx = (mu+self.heatmap_mx*sg)
        for i in range(3):
            x[:,:,i] = np.clip(x[:,:,i], mi[i], mx[i])
            x[:,:,i] = 255*(x[:,:,i] - mi[i]) / (mx[i]-mi[i])
        x = 255-x

        out = self.heatmap_alpha*x + (1-self.heatmap_alpha)*self.tb.img
        out = np.rint(out).astype(np.uint8)
        self.ax.imshow(out)
        self.update_plot()

    def segment_cortex(self):
        heatmap_fpath = os.path.join(self.tmp_directory, 'heatmap.npy')
        if not os.path.exists(heatmap_fpath):
            self.aggregate_cells()
        self.tissue_mask = sana.image.Frame(os.path.join(self.tmp_directory, 'tissue_mask.png'))
        self.heatmap = sana.image.Frame(np.load(heatmap_fpath))

        self.logger.data['mpp'] = self.loader.mpp
        self.logger.data['ds'] = self.loader.ds
        #self.logger.data['ds_thumbnail'] = self.feature_ds
        self.logger.data['ds_thumbnail'] = 1.0
        self.logger.data['thumbnail_level'] = self.tb.level
        gm_mask, wm_mask, self.contours = neuseg.tissue.segment_wm(
            feature_heatmap=self.heatmap,
            tissue_mask=self.tissue_mask,
            tb=self.tb,
            logger=self.logger,)
        gm_mask = sana.image.Frame(gm_mask)
        gm_mask.to_binary()
        self.gm_polys, _ = gm_mask.to_polygons()
        
        self.gm_mask = self.tissue_mask.copy()
        self.gm_mask.img = (self.gm_mask.img / np.max(self.gm_mask.img)).astype(np.uint8)
        self.gm_mask.img += (wm_mask[:,:,None]/np.max(wm_mask)).astype(np.uint8)

        # allow user drawn annotations now that we have all data required
        self.start_tracking()

        self.plot_segmentations()

    def plot_segmentations(self):
        self.ax.clear()
        self.ax.imshow(self.tb.img)
        ds = self.loader.converter.ds[self.tb.level]
        ds = 1
        for x in self.contours['gm_wm']:
            self.ax.plot(*np.array(x).T, color='red')
        for x in self.contours['gm_csf']:
            self.ax.plot(*np.array(x).T, color='blue')
        self.update_plot()

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        for poly in self.gm_polys:
            if sana.geo.ray_tracing(x, y, poly):
                self.loc_x = x
                self.loc_y = y
                self.add_annotation()

    def add_annotation(self):
        if self.loc_x is None:
            return
        l = self.segment_length//(2*self.loader.mpp*self.loader.ds[self.tb.level])
        try:
            csf_poly, csf0, csf1, wm_poly, wm0, wm1 = self.get_roi(self.gm_mask, (self.loc_x, self.loc_y), l=l)
        except:
            return
        csf = csf_poly.slice_shortest(csf0, csf1).astype(float)
        wm = wm_poly.slice_shortest(wm0, wm1).astype(float)
        s0 = sana.geo.curve_like(csf, [csf_poly[csf0][0], wm_poly[wm0][0]], [csf_poly[csf0][1], wm_poly[wm0][1]]).astype(float)
        s1 = sana.geo.curve_like(csf, [csf_poly[csf1][0], wm_poly[wm1][0]], [csf_poly[csf1][1], wm_poly[wm1][1]]).astype(float)
        self.plot_segmentations()
        self.ax.plot(*csf.T, color='pink')
        self.ax.plot(*wm.T, color='yellow')
        self.ax.plot(*s0.T, color='black')
        self.ax.plot(*s1.T, color='black')
        self.update_plot()
        annos = [
            csf.to_annotation('CSF'),
            s0.to_annotation('R'),
            wm.to_annotation('WM'),
            s1.to_annotation('L'),
        ]
        sana.utils.write_geojson(os.path.join(self.tmp_directory, 'annotations.geojson'), annos)

        self.update_plot()

    def get_shortest(self, gm, ctr, n_angles=360, debug=False):

        dists, pts = [], []
        for theta in np.linspace(0, np.pi, n_angles):
            csf, wm = None,None
        
            r = 0
            direction = +1
            while True:
                r += direction
                x = int(r*np.cos(theta) + ctr[0])
                y = int(r*np.sin(theta) + ctr[1])
                if x < 0 or y < 0 or x >= gm.img.shape[1] or y >= gm.img.shape[0]:
                    break            
                if csf is None and gm.img[y,x] == 0:
                    csf = (x,y)
                    break
                if wm is None and gm.img[y,x] == 2:
                    wm = (x,y)
                    break

            r = 0
            direction = -1
            while True:
                r += direction
                x = int(r*np.cos(theta) + ctr[0])
                y = int(r*np.sin(theta) + ctr[1])
                if x < 0 or y < 0 or x >= gm.img.shape[1] or y >= gm.img.shape[0]:
                    break
                if csf is None and gm.img[y,x] == 0:
                    csf = (x,y)
                    break
                if wm is None and gm.img[y,x] == 2:
                    wm = (x,y)
                    break

            if csf is None or wm is None:
                continue
        
            dists.append(np.sqrt((csf[0]-wm[0])**2+(csf[1]-wm[1])**2))
            pts.append([csf,wm])

        return pts[np.argmin(dists)]
        
    def get_roi(self, gm, ctr, l):
        x, y = ctr

        # get the spine of the ROI
        csf, wm = self.get_shortest(gm, (x,y))
    
        # get the lateral direction of cortex
        th = np.arctan2((wm[1]-csf[1]), (wm[0]-csf[0]))
        thp = th + np.pi/2
    
        p0 = (-l*np.cos(thp)+ctr[0],-l*np.sin(thp)+ctr[1])
        p1 = (l*np.cos(thp)+ctr[0],l*np.sin(thp)+ctr[1])

        # get the left and right boundaries
        csf0, wm0 = self.get_shortest(gm, p0)
        csf1, wm1 = self.get_shortest(gm, p1)

        # get the polygon vertices to sample at
        # gm_mask = gm.copy()
        # gm_mask.img[gm_mask.img == 2] = 0
        # gm_polys = gm_mask.to_polygons()[0]
        gm_polys = self.gm_polys
        sample_pts = []
        sample_idxs = []

        dists, csf_idxs = [], []
        for poly in gm_polys:
            d0 = np.sqrt((poly[:,0]-csf0[0])**2+(poly[:,1]-csf0[1])**2)
            d1 = np.sqrt((poly[:,0]-csf1[0])**2+(poly[:,1]-csf1[1])**2)
            csf_idxs.append([np.argmin(d0),np.argmin(d1)])
            dists.append(np.min(d0)+np.min(d1))
        csf_idx_0, csf_idx_1 = csf_idxs[np.argmin(dists)]
        csf_sample_poly = gm_polys[np.argmin(dists)]

        dists, wm_idxs = [], []
        for poly in gm_polys:
            d0 = np.sqrt((poly[:,0]-wm0[0])**2+(poly[:,1]-wm0[1])**2)
            d1 = np.sqrt((poly[:,0]-wm1[0])**2+(poly[:,1]-wm1[1])**2)
            wm_idxs.append([np.argmin(d0),np.argmin(d1)])
            dists.append(np.min(d0)+np.min(d1))
        wm_idx_0, wm_idx_1 = wm_idxs[np.argmin(dists)]
        wm_sample_poly = gm_polys[np.argmin(dists)]

        def slice_shortest(p, i, j, level):
            if j < i:
                (i,j) = (j,i)
            opt1 = p[i:j]
            opt2 = np.concatenate([p[j:],p[:i]])
            opt1 = pdnl_sana.geo.Curve(*opt1.T, is_micron=False, level=level)
            opt2 = pdnl_sana.geo.Curve(*opt2.T, is_micron=False, level=level)        
            if opt1.get_length() < opt2.get_length():
                return opt1
            else:
                return opt2
        #csf_gm = slice_shortest(csf_sample_poly, csf_idx_0, csf_idx_1, level=gm.level)
        #gm_wm = slice_shortest(wm_sample_poly, wm_idx_0, wm_idx_1, level=gm.level)

        #return csf_gm, gm_wm, [[csf,wm],[p0,p1],[csf0,wm0],[csf1,wm1]]
        return csf_sample_poly, csf_idx_0, csf_idx_1, wm_sample_poly, wm_idx_0, wm_idx_1
        
