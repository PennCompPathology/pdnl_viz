
# Penn Digital Neuropathology Lab's Visualization Tools

This project is an ongoing development of visualization tools used for digital IHC analysis. 

## Requirements

Python3.12 or newer: https://github.com/PackeTsar/Install-Python$0


## Demo Installation

Create a virtual environment

Mac/Linux
```bash
python3 -m venv .pdnl_demo_venv
source .pdnl_demo_venv/bin/activate
```
Windows
```bash
python3 -m venv .pdnl_demo_venv
.pdnl_demo_venv\Scripts\activate
```
    
Install pdnl_viz using pip

```bash
python -m pip install pdnl_viz
```


## Usage

```bash
pdnl_demo
```


## Troubleshooting

#### Issues with numba, llvmlite, scikit-fmm

This often has to do with C++ compiler problems. 

On Mac, try this: https://stackoverflow.com/a/34617930

On Windows, try installing the base tools here: https://visualstudio.microsoft.com/visual-cpp-build-tools/$0



## Support

For support, email noah.capp@pennmedicine.upenn.edu


## Demo

