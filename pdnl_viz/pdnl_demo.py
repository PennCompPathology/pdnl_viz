
import os
import sys
import time

os.environ["QT_API"] = "PyQt6"

from PyQt6 import QtCore, QtWidgets


import tempfile
import cv2
import numpy as np
import random

import pdnl_sana as sana
import pdnl_sana.logging
import pdnl_sana.slide
import pdnl_sana.image
import pdnl_sana.geo
import pdnl_sana.process
import pdnl_sana.filter
import pdnl_sana.segment
import pdnl_sana.quantify
import neuseg.tissue
import neuseg.nuclei

from .widgets import ThumbnailWidget, DABWidget, CSWidget, NeusegWidget, OverlayWidget, ROIWidget

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

class MainWindow(QtWidgets.QMainWindow):
    RNG_SPIN_STEP = 0.5
    RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "resources")

    def __init__(self, tmp_directory=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tmp_directory is None:
            self.temp_directory = tempfile.TemporaryDirectory()
            self.tmp_directory = self.temp_directory.name
        else:        
            self.tmp_directory = tmp_directory

        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)

        self.main_layout = QtWidgets.QHBoxLayout()
        self.widget.setLayout(self.main_layout)

        self.thumbnail_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.thumbnail_layout)

        self.slide_layout = QtWidgets.QHBoxLayout()
        self.thumbnail_layout.addLayout(self.slide_layout)

        self.tau_button = QtWidgets.QPushButton('Tau Pathology (AT8)')
        self.slide_layout.addWidget(self.tau_button)        
        self.tau_button.clicked.connect(self.load_tau_slide)

        self.neuron_button = QtWidgets.QPushButton('Neuronal (SMI32)')
        self.slide_layout.addWidget(self.neuron_button)
        self.neuron_button.clicked.connect(self.load_neuron_slide)
        self.neuron_button.setEnabled(False)
        
        self.button_layout = QtWidgets.QHBoxLayout()
        self.thumbnail_layout.addLayout(self.button_layout)

        self.dab_button = QtWidgets.QPushButton('DAB Processing')
        self.dab_button.pressed.connect(self.show_dab)
        self.dab_button.setToolTip("""
        Optionally apply smoothing and background normalization, then select a threshold semi-automatically by setting your desired strictness.
        """)
        self.overlay_button = QtWidgets.QPushButton('DAB Overlay')
        self.overlay_button.pressed.connect(self.show_overlay)
        self.overlay_button.setToolTip("""
        Processes the Whole Slide Image using the parameters selected in \"DAB Processing\". The DAB is smoothed using a Gaussian window.
        """)
        self.cs_button = QtWidgets.QPushButton('Counterstain Processing')
        self.cs_button.pressed.connect(self.show_cs)
        self.cs_button.setToolTip("""
        Optionally apply smoothing and background normalization, then select a threshold semi-automatically by setting your desired strictness. Morphology filters help to control noise.
        """)
        self.neuseg_button = QtWidgets.QPushButton('NEUSEG Demo')
        self.neuseg_button.pressed.connect(self.show_neuseg)
        self.neuseg_button.setToolTip("""
        Generate Cortical Segmentations utilizing parameters selected in \"Counterstain Processing\"
        """)
        self.roi_button = QtWidgets.QPushButton('ROI Analysis')
        self.roi_button.pressed.connect(self.show_roi)
        self.roi_button.setToolTip("""
        Quantify DAB and Counterstain features within the selected ROI
        """)
        self.button_layout.addWidget(self.dab_button)
        self.button_layout.addWidget(self.overlay_button)
        self.button_layout.addWidget(self.cs_button)
        self.button_layout.addWidget(self.neuseg_button)
        self.button_layout.addWidget(self.roi_button)

        self.thumbnail_widget = ThumbnailWidget(self, width=5, height=4)
        self.thumbnail_widget.updated.connect(self.update_frame)
        self.thumbnail_layout.addWidget(self.thumbnail_widget)

        self.interactive_layout = None
        self.dab_widget = None
        self.cs_widget = None

        self.show()

        self.automate_start()

    # Source - https://stackoverflow.com/a/13790741
    # Posted by max, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-31, License - CC BY-SA 3.0
    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def automate_start(self):
        pass
        self.showMaximized()
        self.tau_button.click()
        #self.roi_button.click()
        #self.overlay_button.click()
        #self.neuseg_button.click()
        #self.update()
        #self.cortex_button.click()

    def load_slide(self, slide_path, staining_code):
        self.slide_path = slide_path
        self.slide_name = os.path.splitext(os.path.basename(self.slide_path))[0]
        self.logger = pdnl_sana.logging.Logger('debug', 'x.pkl')
        self.loader = pdnl_sana.slide.Loader(self.logger, self.slide_path)
        self.staining_code = staining_code
        self.thumbnail_widget.set_slide(self.loader)

    def load_tau_slide(self):
        slide_path = os.path.join(self.resource_path('.'), self.RESOURCE_PATH, 'tau.tif')
        self.load_slide(slide_path, 'HDAB')
    def load_neuron_slide(self):
        slide_path = None
        self.load_slide(slide_path, 'HDAB')
    def update_frame(self):
        if not self.dab_widget is None:
            loc, size = self.thumbnail_widget.get_rect()
            self.dab_widget.set_slide(self.loader, loc, size)
        if not self.cs_widget is None:
            loc, size = self.thumbnail_widget.get_rect()
            self.cs_widget.set_slide(self.loader, loc, size)

    def reset_interactive(self):
        if not self.interactive_layout is None:
            self.delete_boxlayout(self.main_layout, self.interactive_layout)
            self.interactive_layout = None
            self.dab_widget = None
            self.cs_widget = None
            self.neuseg_widget = None
            self.overlay_widget = None
        self.interactive_layout = QtWidgets.QVBoxLayout()

    def show_roi(self):
        self.reset_interactive()
        self.main_layout.addLayout(self.interactive_layout)        

        self.roi_widget = ROIWidget(self, width=5, height=4, anno_file=os.path.join(self.tmp_directory, 'annotations.geojson'))
        self.interactive_layout.addWidget(self.roi_widget)

        self.parameters_layout = QtWidgets.QHBoxLayout()
        self.interactive_layout.addLayout(self.parameters_layout)

        self.nlayers_spin = QtWidgets.QSpinBox()
        self.parameters_layout.addWidget(QtWidgets.QLabel('Num. Layers'))
        self.parameters_layout.addWidget(self.nlayers_spin)
        self.nlayers_spin.setRange(2, 100)
        self.nlayers_spin.setValue(50)
        self.nlayers_spin.setSingleStep(10)
        self.nlayers_spin.valueChanged.connect(self.update_roi_parameters)
        self.nlayers_spin.setToolTip("""
        Defines the resolution for quantifying digital layers of the signal
        """)
        
        self.roi_widget.set_slide(self.loader)
        self.roi_widget.set_deform()
        self.roi_widget.process_stains()

    def show_dab(self):
        self.reset_interactive()

        self.dab_widget = DABWidget(self, width=5, height=4)
        self.update_frame()
        self.interactive_layout.addWidget(self.dab_widget, stretch=10)
        self.main_layout.addLayout(self.interactive_layout)

        self.add_thresholding_widgets()
        
        self.update_stain()

    def show_cs(self):
        self.reset_interactive()

        self.cs_widget = CSWidget(self, width=5, height=4)
        self.update_frame()
        self.interactive_layout.addWidget(self.cs_widget, stretch=10)
        self.main_layout.addLayout(self.interactive_layout)

        self.add_thresholding_widgets()
        self.add_morphology_widgets()
        self.add_segment_widgets()

        self.update_stain()

    def show_overlay(self):
        self.reset_interactive()
        self.main_layout.addLayout(self.interactive_layout)

        self.overlay_widget = OverlayWidget(self, width=5, height=4, tmp_directory=self.tmp_directory)
        self.interactive_layout.addWidget(self.overlay_widget, stretch=10)

        self.parameters_layout = QtWidgets.QHBoxLayout()
        self.interactive_layout.addLayout(self.parameters_layout)

        self.preprocess_button = QtWidgets.QPushButton('Preprocess Chunks')
        self.parameters_layout.addWidget(self.preprocess_button)        
        self.preprocess_button.clicked.connect(self.overlay_widget.process_dab)
        self.preprocess_button.setToolTip("""
        Color Deconvolution and Global Threshold Calculation
        """)
        
        self.window_spin = QtWidgets.QSpinBox()
        self.parameters_layout.addWidget(QtWidgets.QLabel('Window Size (microns)'))
        self.parameters_layout.addWidget(self.window_spin)
        self.window_spin.setMinimum(10)
        self.window_spin.setMaximum(200)
        self.window_spin.setSingleStep(20)
        self.window_spin.setValue(50)
        self.window_spin.valueChanged.connect(self.update_overlay_aggregate_parameters)
        self.window_spin.setToolTip("""
        Amount of smoothing to apply to the WSI DAB detections
        """)

        self.alpha_spin = QtWidgets.QDoubleSpinBox()
        self.parameters_layout.addWidget(QtWidgets.QLabel('Alpha'))
        self.parameters_layout.addWidget(self.alpha_spin)
        self.alpha_spin.setMinimum(0.1)
        self.alpha_spin.setMaximum(1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.valueChanged.connect(self.update_overlay_plot_parameters)
        self.alpha_spin.setToolTip("""
        Transparency for the heatmap
        """)

        self.overlay_widget.set_slide(self.loader, self.staining_code)
        self.overlay_widget.plot_dab(reprocess=True)

    def add_thresholding_widgets(self):
        self.thresh_layout = QtWidgets.QHBoxLayout()
        self.interactive_layout.addLayout(self.thresh_layout, stretch=1)

        self.thresh_layout.addWidget(QtWidgets.QLabel('Strictness'))
        self.strictness_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.thresh_layout.addWidget(self.strictness_slider)
        self.strictness_delta = 0.05
        self.strictness_mi = -0.95
        self.strictness_mx = 0.95
        self.strictness_rng = self.strictness_mx - self.strictness_mi + self.strictness_delta
        self.strictness_slider.setRange(0, int(self.strictness_rng // self.strictness_delta))
        self.strictness_slider.setValue(self.strictness_slider.maximum()//2)
        self.strictness_slider.valueChanged.connect(self.update_threshold)
        self.strictness_slider.setToolTip("""
        Higher strictness clips the left side of the histogram, reducing amount of pixels which will pass through the threshold. Lower strictness clips the right side of the histogram.
        """)

        self.smoothing_layout = QtWidgets.QHBoxLayout()
        self.interactive_layout.addLayout(self.smoothing_layout, stretch=1)
        self.smoothing_checkbox = QtWidgets.QCheckBox('Smoothing')
        self.smoothing_layout.addWidget(self.smoothing_checkbox)
        self.smoothing_checkbox.setChecked(False)
        self.smoothing_checkbox.stateChanged.connect(self.update_stain)
        self.smoothing_checkbox.setToolTip("""Anistrophic Smoothing preserves the boundaries of objects while equalizing the interiors. Useful for Object Detection.
        """)

        self.smoothing_layout.addStretch()

        self.background_layout = QtWidgets.QHBoxLayout()
        self.interactive_layout.addLayout(self.background_layout, stretch=1)        
        self.background_checkbox = QtWidgets.QCheckBox('Background Subtraction')
        self.background_layout.addWidget(self.background_checkbox)        
        self.background_checkbox.setChecked(False)
        self.background_checkbox.stateChanged.connect(self.update_stain)
        self.background_checkbox.setToolTip("""
        Calculates and subtracts the background staining from the Color Deconvolution
        """)

        self.background_radius_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        # self.background_layout.addWidget(QtWidgets.QLabel('Radius'))
        # self.background_layout.addWidget(self.background_radius_slider)
        self.radius_delta = 20            
        self.background_radius_slider.setMinimum(0)
        self.background_radius_slider.setMaximum((200-self.radius_delta)//self.radius_delta)
        self.background_radius_slider.setValue(self.background_radius_slider.maximum()//2)
        self.background_radius_slider.valueChanged.connect(self.update_stain)

    def add_morphology_widgets(self):
        self.closing_layout = QtWidgets.QHBoxLayout()
        self.interactive_layout.addLayout(self.closing_layout, stretch=1)

        self.closing_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.closing_layout.addWidget(QtWidgets.QLabel('Morphology Closing'))
        self.closing_layout.addWidget(self.closing_slider)
        self.closing_slider.setMaximum(10)
        self.closing_slider.setValue(0)
        self.closing_slider.valueChanged.connect(self.update_threshold)
        self.closing_slider.setToolTip("""
        This filter closes holes within objects which pass through the threshold.
        """)

        self.opening_layout = QtWidgets.QHBoxLayout()
        self.interactive_layout.addLayout(self.opening_layout, stretch=1)

        self.opening_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.opening_layout.addWidget(QtWidgets.QLabel('Morphology Opening'))
        self.opening_layout.addWidget(self.opening_slider)
        self.opening_slider.setMaximum(10)
        self.opening_slider.setValue(0)
        self.opening_slider.valueChanged.connect(self.update_threshold)
        self.opening_slider.setToolTip("""
        This filter deletes small objects which pass through the threshold.
        """)        

    # TODO: add minimum soma radius etc
    def add_segment_widgets(self):
        pass

    def show_neuseg(self):
        self.reset_interactive()

        self.neuseg_widget = NeusegWidget(self, width=5, height=4, tmp_directory=self.tmp_directory)
        self.interactive_layout.addWidget(self.neuseg_widget, stretch=10)
        self.main_layout.addLayout(self.interactive_layout)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.interactive_layout.addLayout(self.button_layout, stretch=1)

        self.preprocess_button = QtWidgets.QPushButton('Preprocess Chunks')
        self.button_layout.addWidget(self.preprocess_button)        
        self.preprocess_button.clicked.connect(self.neuseg_widget.preprocess_cs)
        self.preprocess_button.setToolTip("""
        Color Deconvolution and Global Threshold Calculation
        """)

        self.segcells_button = QtWidgets.QPushButton('Segment Cells')
        self.button_layout.addWidget(self.segcells_button)
        self.segcells_button.clicked.connect(self.neuseg_widget.segment_cells)
        self.segcells_button.setToolTip("""
        Instance segmentation of the surviving objects from Counterstain Thresholding
        """)

        self.heatmap_button = QtWidgets.QPushButton('Aggregate Cells')
        self.button_layout.addWidget(self.heatmap_button)
        self.heatmap_button.clicked.connect(self.neuseg_widget.aggregate_cells)
        self.heatmap_button.setToolTip("""
        Creates a heatmap of counterstain cell features -- Density, Avg. Area, & Avg. Intensity
        """)

        self.cortex_button = QtWidgets.QPushButton('Segment Cortex')
        self.button_layout.addWidget(self.cortex_button)
        self.cortex_button.clicked.connect(self.neuseg_widget.segment_cortex)
        self.cortex_button.setToolTip("""
        Classifies pixels as White Matter or Gray Matter utilizing a Gaussian Mixture Model trained on the Cell Feature Heatmap.
        """)

        self.parameters_layout = QtWidgets.QVBoxLayout()
        self.interactive_layout.addLayout(self.parameters_layout, stretch=1)

        l = QtWidgets.QHBoxLayout()
        self.parameters_layout.addLayout(l)

        self.window_spin = QtWidgets.QSpinBox()
        l.addWidget(QtWidgets.QLabel('Heatmap Window'))
        l.addWidget(self.window_spin)
        self.window_spin.setRange(100, 2000)
        self.window_spin.setValue(1000)
        self.window_spin.setSingleStep(250)
        self.window_spin.valueChanged.connect(self.update_neuseg_parameters)
        self.window_spin.setToolTip("""
        Amount of smoothing to apply to the Cell Features
        """)

        self.alpha_spin = QtWidgets.QDoubleSpinBox()
        l.addWidget(QtWidgets.QLabel('Heatmap Transparency'))
        l.addWidget(self.alpha_spin)
        self.alpha_spin.setMinimum(0.0)
        self.alpha_spin.setMaximum(1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.valueChanged.connect(self.update_neuseg_parameters)
        self.alpha_spin.setToolTip("Transparency")
        
        self.mi_spins = []
        self.mx_spins = []
        names = ['Cell Density', 'Cell Area', 'Cell Intensity']
        for i in range(3):
            layout = QtWidgets.QHBoxLayout()
            self.parameters_layout.addLayout(layout)

            x = QtWidgets.QDoubleSpinBox()
            layout.addWidget(QtWidgets.QLabel(f'{names[i]} Range'))
            layout.addWidget(x)
            self.mi_spins.append(x)
            x.setMinimum(-10.0)
            x.setValue(-1.0)
            x.setSingleStep(self.RNG_SPIN_STEP)
            x.valueChanged.connect(self.update_neuseg_parameters)
            x.setToolTip("""
            Num. of Stddevs below the mean
            """)

            x = QtWidgets.QDoubleSpinBox()
            layout.addWidget(x)
            self.mx_spins.append(x)
            x.setValue(+1.0)
            x.setSingleStep(self.RNG_SPIN_STEP)
            x.valueChanged.connect(self.update_neuseg_parameters)
            x.setToolTip("""
            Num. of Stddevs above the mean
            """)
        
        self.length_spin = QtWidgets.QSpinBox()
        self.interactive_layout.addWidget(QtWidgets.QLabel('Cortical Segmentation Length (microns)'))
        self.interactive_layout.addWidget(self.length_spin)
        self.length_spin.setRange(100, 3000)
        self.length_spin.setValue(1000)
        self.length_spin.setSingleStep(500)
        self.length_spin.valueChanged.connect(self.update_neuseg_segment_parameters)
        self.length_spin.setToolTip("""
        Distance across the cortex to sample. The resulting ROI will always span from the edge of the tissue to the WM.
        """)

        self.neuseg_widget.set_slide(self.loader, self.staining_code)

    def update_overlay_aggregate_parameters(self):
        self.overlay_widget.window_size = self.window_spin.value()
        self.overlay_widget.aggregate_dab()
        self.overlay_widget.plot_dab()

    def update_overlay_plot_parameters(self):
        self.overlay_widget.alpha = self.alpha_spin.value()
        self.overlay_widget.plot_dab()

    def update_roi_parameters(self):
        self.roi_widget.nlayers = self.nlayers_spin.value()
        self.roi_widget.set_layer_masks()
        self.roi_widget.plot_dab_curves()
        self.roi_widget.plot_cs_curves()

    def update_neuseg_parameters(self):
        for i in range(3):
            self.neuseg_widget.heatmap_mi[i] = self.mi_spins[i].value()
            self.neuseg_widget.heatmap_mx[i] = self.mx_spins[i].value()
        self.neuseg_widget.window_size = self.window_spin.value()
        self.neuseg_widget.heatmap_alpha = self.alpha_spin.value()
        self.neuseg_widget.plot_heatmap()
    def update_neuseg_segment_parameters(self):
        self.neuseg_widget.segment_length = self.length_spin.value()
        self.neuseg_widget.add_annotation()

    def update_thresholding_parameters(self):
        if not self.dab_widget is None:
            active = self.dab_widget
        else:
            active = self.cs_widget
        active.apply_smoothing = self.smoothing_checkbox.isChecked()
        active.normalize_background = self.background_checkbox.isChecked()
        active.strictness = (self.strictness_slider.value()+1) * self.strictness_delta + self.strictness_mi
        active.background_radius = (self.background_radius_slider.value()+1)*self.radius_delta

    def update_morphology_parameters(self):
        if not self.dab_widget is None:
            active = self.dab_widget
        else:
            active = self.cs_widget
        active.closing_radius = self.closing_slider.value()
        active.opening_radius = self.opening_slider.value()        

    def update_stain(self):
        self.update_thresholding_parameters()
        if not self.dab_widget is None:
            self.dab_widget.set_processor()
        else:
            self.cs_widget.set_processor()

    def update_threshold(self):
        self.update_thresholding_parameters()
        if not self.dab_widget is None:
            self.dab_widget.set_threshold()
        else:
            self.update_morphology_parameters()
            self.cs_widget.set_threshold()
    
    def delete_items_of_layout(self, layout):
        if not layout is None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if not widget is None:
                    widget.setParent(None)
                else:
                    self.delete_items_of_layout(item.layout())
    def delete_boxlayout(self, parent, layout):
        for i in range(parent.count()):
            layout_item = parent.itemAt(i)
            if layout_item.layout() == layout:
                self.delete_items_of_layout(layout_item.layout())
                parent.removeItem(layout_item)
                break

def main():
    app = QtWidgets.QApplication(sys.argv)
    #tmp_directory = './test_tmp'
    tmp_directory = None
    window = MainWindow(tmp_directory=tmp_directory)
    app.exec()

if __name__ == "__main__":
    main()


