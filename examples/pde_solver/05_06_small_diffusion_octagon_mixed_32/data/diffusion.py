# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`
import time
import os
import sys
import numpy as np
#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
sim_id = sys.argv[1]
pixels = 32
thepath = 'THEPATH'
thepath = '/home/xiaoxuan/globus/2-PDE-ss/data/diffusion/small-32x32-octagon-1bvp'

# create a new 'Legacy VTK Reader'
output1vtk = LegacyVTKReader(FileNames=[thepath + '/output-1.vtk'])

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1081, 835]

# show data in view
output1vtkDisplay = Show(output1vtk, renderView1)

# get color transfer function/color map for 'c'
lithiumLUT = GetColorTransferFunction('c')

# get opacity transfer function/opacity map for 'c'
lithiumPWF = GetOpacityTransferFunction('c')

# trace defaults for the display properties.
output1vtkDisplay.Representation = 'Surface'
output1vtkDisplay.AmbientColor = [0.0, 0.0, 0.0]
output1vtkDisplay.ColorArrayName = ['POINTS', 'c']
output1vtkDisplay.LookupTable = lithiumLUT
output1vtkDisplay.OSPRayScaleArray = 'c'
output1vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
output1vtkDisplay.SelectOrientationVectors = 'c'
output1vtkDisplay.ScaleFactor = 0.1
output1vtkDisplay.SelectScaleArray = 'c'
output1vtkDisplay.GlyphType = 'Arrow'
output1vtkDisplay.GlyphTableIndexArray = 'c'
output1vtkDisplay.GaussianRadius = 0.005
output1vtkDisplay.SetScaleArray = ['POINTS', 'c']
output1vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
output1vtkDisplay.OpacityArray = ['POINTS', 'c']
output1vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
output1vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
output1vtkDisplay.SelectionCellLabelFontFile = ''
output1vtkDisplay.SelectionPointLabelFontFile = ''
output1vtkDisplay.PolarAxes = 'PolarAxesRepresentation'
output1vtkDisplay.ScalarOpacityFunction = lithiumPWF
output1vtkDisplay.ScalarOpacityUnitDistance = 0.09969186880802686

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
output1vtkDisplay.OSPRayScaleFunction.Points = [-0.38565274439484715, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
output1vtkDisplay.ScaleTransferFunction.Points = [-0.38565274439484715, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
output1vtkDisplay.OpacityTransferFunction.Points = [-0.38565274439484715, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
output1vtkDisplay.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
output1vtkDisplay.DataAxesGrid.XTitleFontFile = ''
output1vtkDisplay.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
output1vtkDisplay.DataAxesGrid.YTitleFontFile = ''
output1vtkDisplay.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
output1vtkDisplay.DataAxesGrid.ZTitleFontFile = ''
output1vtkDisplay.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
output1vtkDisplay.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
output1vtkDisplay.DataAxesGrid.XLabelFontFile = ''
output1vtkDisplay.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
output1vtkDisplay.DataAxesGrid.YLabelFontFile = ''
output1vtkDisplay.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
output1vtkDisplay.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
output1vtkDisplay.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
output1vtkDisplay.PolarAxes.PolarAxisTitleFontFile = ''
output1vtkDisplay.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
output1vtkDisplay.PolarAxes.PolarAxisLabelFontFile = ''
output1vtkDisplay.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
output1vtkDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
output1vtkDisplay.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
output1vtkDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.CameraPosition = [0.5, 0.5396810000000001, 10000.0]
renderView1.CameraFocalPoint = [0.5, 0.5396810000000001, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
output1vtkDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator1 = Calculator(Input=output1vtk)
calculator1.Function = ''

# Properties modified on calculator1
calculator1.Function = 'c*0.5+0.5'

# show data in view
calculator1Display = Show(calculator1, renderView1)

# get color transfer function/color map for 'Result'
resultLUT = GetColorTransferFunction('Result')

# get opacity transfer function/opacity map for 'Result'
resultPWF = GetOpacityTransferFunction('Result')

# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'
calculator1Display.AmbientColor = [0.0, 0.0, 0.0]
calculator1Display.ColorArrayName = ['POINTS', 'Result']
calculator1Display.LookupTable = resultLUT
calculator1Display.OSPRayScaleArray = 'Result'
calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1Display.SelectOrientationVectors = 'c'
calculator1Display.ScaleFactor = 0.1
calculator1Display.SelectScaleArray = 'Result'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'Result'
calculator1Display.GaussianRadius = 0.005
calculator1Display.SetScaleArray = ['POINTS', 'Result']
calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1Display.OpacityArray = ['POINTS', 'Result']
calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1Display.SelectionCellLabelFontFile = ''
calculator1Display.SelectionPointLabelFontFile = ''
calculator1Display.PolarAxes = 'PolarAxesRepresentation'
calculator1Display.ScalarOpacityFunction = resultPWF
calculator1Display.ScalarOpacityUnitDistance = 0.09969186880802686

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
calculator1Display.OSPRayScaleFunction.Points = [-0.38565274439484715, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [-0.38565274439484715, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [-0.38565274439484715, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
calculator1Display.DataAxesGrid.XTitleColor = [0.0, 0.0, 0.0]
calculator1Display.DataAxesGrid.XTitleFontFile = ''
calculator1Display.DataAxesGrid.YTitleColor = [0.0, 0.0, 0.0]
calculator1Display.DataAxesGrid.YTitleFontFile = ''
calculator1Display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
calculator1Display.DataAxesGrid.ZTitleFontFile = ''
calculator1Display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
calculator1Display.DataAxesGrid.XLabelColor = [0.0, 0.0, 0.0]
calculator1Display.DataAxesGrid.XLabelFontFile = ''
calculator1Display.DataAxesGrid.YLabelColor = [0.0, 0.0, 0.0]
calculator1Display.DataAxesGrid.YLabelFontFile = ''
calculator1Display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
calculator1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
calculator1Display.PolarAxes.PolarAxisTitleColor = [0.0, 0.0, 0.0]
calculator1Display.PolarAxes.PolarAxisTitleFontFile = ''
calculator1Display.PolarAxes.PolarAxisLabelColor = [0.0, 0.0, 0.0]
calculator1Display.PolarAxes.PolarAxisLabelFontFile = ''
calculator1Display.PolarAxes.LastRadialAxisTextColor = [0.0, 0.0, 0.0]
calculator1Display.PolarAxes.LastRadialAxisTextFontFile = ''
calculator1Display.PolarAxes.SecondaryRadialAxesTextColor = [0.0, 0.0, 0.0]
calculator1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# hide data in view
Hide(output1vtk, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

coors = np.linspace(0,1.0,pixels)
# print(coors, len(coors))
# exit(0)

# create a new 'Probe Location'

probeLocation1 = ProbeLocation(Input=calculator1, ProbeType='Fixed Radius Point Source')
count = 0
for x in coors:
    for y in coors:
        #print(sim_id, x, y, pixels*pixels-count)
        # t_start = time.time()
        # print('0:', time.time()-t_start)
        # print('a:', time.time()-t_start)
        
        # init the 'Fixed Radius Point Source' selected for 'ProbeType'
        probeLocation1.ProbeType.Center = [x, y, 0.0]
        # print('b:', time.time()-t_start)
        
        # save data
        SaveData(thepath + '/p'+str(count)+'.csv', proxy=probeLocation1)
        # print('c:', time.time()-t_start)
        # del probeLocation1 
        # print('d:', time.time()-t_start)
        count += 1

# check the csv file to see which one to load
def read_value (file_name):
    with open(file_name, 'r') as f:
        last_line = f.readlines()[-1]
    # results, c
    value = last_line.split(',')[0]
    return value

results = np.zeros((pixels,pixels))
count = 0
for x in range(0, pixels):
    for y in range(0, pixels):
        values = read_value(thepath + '/p'+str(count)+'.csv')
        results[pixels-1-y][x] = values
        count += 1

results = np.where(results == 0, -1, results )
# results = ma.masked_where(results == -1, results )
# fig, axs = plt.subplots(1, 1)
# axs.imshow(results)
# plt.show()
results = np.expand_dims(results, axis=0)
results = np.expand_dims(results, axis=3)
np.save(thepath + "/np-labels-"+str(sim_id)+".npy", results)

os.system('')
# os.system('cd /home/xiaoxuan/Documents/2-Codes/9_ssb/Example10_diffusion_steady_state/build/diffusion/0 && zip csv.zip p*.csv ')
os.system('cd ' + thepath + ' && tar -czvf csv.tar.gz p*.csv  ')
os.system('rm -rf ' + thepath + '/p*.csv ')
