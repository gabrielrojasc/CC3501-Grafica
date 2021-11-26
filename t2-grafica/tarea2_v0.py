# coding=utf-8
"""
Tarea 2

Modelo elegido: avion 1
"""

from OpenGL.GL.ARB import transform_feedback_instanced, vertex_attrib_64bit
import glfw
import math
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path

from numpy.core.fromnumeric import trace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath

__author__ = "Gabriel Rojas"
__license__ = "MIT"


# A class to store the application control
class Controller:

  def __init__(self):
    self.fillPolygon = True
    self.showAxis = True
    self.viewPos = np.array([10, 10, 10])
    self.camUp = np.array([0, 1, 0])
    self.distance = 10


controller = Controller()


def setPlot(pipeline, mvpPipeline):
  projection = tr.perspective(45, float(width) / float(height), 0.1, 100)

  glUseProgram(mvpPipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE,
      projection)

  glUseProgram(pipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE,
      projection)

  glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
  glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
  glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

  glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
  glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
  glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

  glUniform3f(
      glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 5, 5, 5)

  glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 1000)
  glUniform1f(
      glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"),
      0.001)
  glUniform1f(
      glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.1)
  glUniform1f(
      glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"),
      0.01)


def setView(pipeline, mvpPipeline):
  view = tr.lookAt(controller.viewPos, np.array([0, 0, 0]), controller.camUp)

  glUseProgram(mvpPipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

  glUseProgram(pipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
  glUniform3f(
      glGetUniformLocation(pipeline.shaderProgram, "viewPosition"),
      controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])


def on_key(window, key, scancode, action, mods):

  if action != glfw.PRESS:
    return

  global controller

  if key == glfw.KEY_SPACE:
    controller.fillPolygon = not controller.fillPolygon

  elif key == glfw.KEY_LEFT_CONTROL:
    controller.showAxis = not controller.showAxis

  elif key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
    glfw.set_window_should_close(window, True)

  elif key == glfw.KEY_1:
    controller.viewPos = np.array(
        [controller.distance, controller.distance,
         controller.distance])  #Vista diagonal 1
    controller.camUp = np.array([0, 1, 0])

  elif key == glfw.KEY_2:
    controller.viewPos = np.array([0, 0, controller.distance])  #Vista frontal
    controller.camUp = np.array([0, 1, 0])

  elif key == glfw.KEY_3:
    controller.viewPos = np.array([controller.distance, 0,
                                   controller.distance])  #Vista lateral
    controller.camUp = np.array([0, 1, 0])

  elif key == glfw.KEY_4:
    controller.viewPos = np.array([0, controller.distance, 0])  #Vista superior
    controller.camUp = np.array([1, 0, 0])

  elif key == glfw.KEY_5:
    controller.viewPos = np.array(
        [controller.distance, controller.distance,
         -controller.distance])  #Vista diagonal 2
    controller.camUp = np.array([0, 1, 0])

  elif key == glfw.KEY_6:
    controller.viewPos = np.array(
        [-controller.distance, controller.distance,
         -controller.distance])  #Vista diagonal 2
    controller.camUp = np.array([0, 1, 0])

  elif key == glfw.KEY_7:
    controller.viewPos = np.array(
        [-controller.distance, controller.distance,
         controller.distance])  #Vista diagonal 2
    controller.camUp = np.array([0, 1, 0])

  else:
    print('Unknown key')


def createGPUShape(pipeline, shape):
  gpuShape = es.GPUShape().initBuffers()
  pipeline.setupVAO(gpuShape)
  gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

  return gpuShape


#NOTA: Aqui creas tu escena. En escencia, sólo tendrías que modificar esta función.
def createScene(pipeline):
  # Defining pi and basic primitives
  pi = math.pi
  r, g, b = np.array([255, 255, 50]) / 256
  sphere = createGPUShape(pipeline, bs.createColorSphereTarea2(r, g, b))
  cone = createGPUShape(pipeline, bs.createColorConeTarea2(r, g, b))
  cube = createGPUShape(pipeline, bs.createColorCubeTarea2(r, g, b))
  cylinder = createGPUShape(pipeline, bs.createColorCylinderTarea2(r, g, b))

  # Construction of the landing gear
  # Create a wheel
  wheelNode = sg.SceneGraphNode('wheel')
  wheelNode.transform = tr.matmul(
      [tr.rotationX(pi / 2), tr.scale(0.5, 0.1, 0.5)])
  wheelNode.childs += [cylinder]

  # Create a landign gear leg
  landingLegsNode = sg.SceneGraphNode('landingLegs')
  landingLegsNode.transform = tr.matmul([
      tr.scale(1, 0.8, 1),
      tr.translate(-0.5, 1.2, 0),
      tr.uniformScale(0.5),
      tr.shearing(0, 0.6, 0, 0, 0, 0),
      tr.rotationZ(pi / 4.5),
      tr.scale(1, 1.5, 0),
      tr.shearing(0, pi / 4, 0, 0, 0, 0)
  ])
  landingLegsNode.childs += [cone]

  # Position the left landing gear leg
  rightLandingLegNode = sg.SceneGraphNode('rightLandingLeg')
  rightLandingLegNode.transform = tr.matmul([tr.translate(0, 0, -0.1)])
  rightLandingLegNode.childs += [landingLegsNode]

  # Position the right landing gear leg
  leftLandingLegNode = sg.SceneGraphNode('leftLandingLeg')
  leftLandingLegNode.transform = tr.matmul([tr.translate(0, 0, 0.1)])
  leftLandingLegNode.childs += [landingLegsNode]

  # Group and position the right wheel and landing gear leg
  rightWheelNode = sg.SceneGraphNode('rightWheel')
  rightWheelNode.transform = tr.matmul([tr.translate(0, 0, 0.7)])
  rightWheelNode.childs += [wheelNode, rightLandingLegNode]

  # Group and position the left wheel and landing gear leg
  leftWheelNode = sg.SceneGraphNode('leftWheel')
  leftWheelNode.transform = tr.matmul([tr.translate(0, 0, -0.7)])
  leftWheelNode.childs += [wheelNode, leftLandingLegNode]

  # Create the rod connecting both wheels
  rodNode = sg.SceneGraphNode('rod')
  rodNode.transform = tr.matmul([tr.rotationX(pi / 2), tr.scale(0.1, 0.7, 0.1)])
  rodNode.childs += [cylinder]

  # Group the langing gear elements and scale
  landingGearNode = sg.SceneGraphNode('landingGear')
  landingGearNode.transform = tr.matmul([tr.uniformScale(0.8)])
  landingGearNode.childs += [rightWheelNode, leftWheelNode, rodNode]

  # Wings
  # Create a wing
  wingNode = sg.SceneGraphNode('wing')
  wingNode.transform = tr.matmul([tr.scale(1, 0.08, 3.5)])
  wingNode.childs += [cube]

  # Position the top wing
  topWingNode = sg.SceneGraphNode('topWing')
  topWingNode.transform = tr.matmul([tr.translate(0, 2.3, 0)])
  topWingNode.childs += [wingNode]

  # Position the bottom wing
  bottomWingNode = sg.SceneGraphNode('bottomWing')
  bottomWingNode.transform = tr.matmul([tr.translate(0, 1, 0)])
  bottomWingNode.childs += [wingNode]

  # Create the wing circle
  wingCircleNode = sg.SceneGraphNode('wingCirle')
  wingCircleNode.transform = tr.matmul([tr.scale(0.9, 0.05, 0.9)])
  wingCircleNode.childs += [cylinder]

  # Postition the right wing circle
  rightWingCirleNode = sg.SceneGraphNode('rightWingCircle')
  rightWingCirleNode.transform = tr.matmul([tr.translate(0, 2.375, 2.8)])
  rightWingCirleNode.childs += [wingCircleNode]

  # Postiton the left wing circle
  leftWingCirleNode = sg.SceneGraphNode('leftWingCircle')
  leftWingCirleNode.transform = tr.matmul([tr.translate(0, 2.375, -2.8)])
  leftWingCirleNode.childs += [wingCircleNode]

  # Create and position the right top wing rounder
  rightTopWingRounderNode = sg.SceneGraphNode('rightTopWingRounder')
  rightTopWingRounderNode.transform = tr.matmul(
      [tr.translate(0, 2.3, 3.5),
       tr.scale(1, 0.08, 1)])
  rightTopWingRounderNode.childs += [cylinder]

  # Create and position the right bottom wing rounder
  rightBottomWingRounderNode = sg.SceneGraphNode('rightBottomWingRounder')
  rightBottomWingRounderNode.transform = tr.matmul(
      [tr.translate(0, 1, 3.5), tr.scale(1, 0.08, 1)])
  rightBottomWingRounderNode.childs += [cylinder]

  # Create and position the left top wing rounder
  leftTopWingRounderNode = sg.SceneGraphNode('leftTopWingRounder')
  leftTopWingRounderNode.transform = tr.matmul(
      [tr.translate(0, 2.3, -3.5),
       tr.scale(1, 0.08, 1)])
  leftTopWingRounderNode.childs += [cylinder]

  # Create and position the left bottom wing rounder
  leftBottomWingRounderNode = sg.SceneGraphNode('leftBottomWingRounder')
  leftBottomWingRounderNode.transform = tr.matmul(
      [tr.translate(0, 1, -3.5), tr.scale(1, 0.08, 1)])
  leftBottomWingRounderNode.childs += [cylinder]

  # Create a wing rod
  wingRodNode = sg.SceneGraphNode('windRod')
  wingRodNode.transform = tr.matmul(
      [tr.scale(0.03, 0.7, 0.03),
       tr.rotationY(pi / 2)])
  wingRodNode.childs += [cylinder]

  # Position the right front wing rod
  wingRodNode1 = sg.SceneGraphNode('wingRod1')
  wingRodNode1.transform = tr.matmul([tr.translate(0.3, 1.65, 3)])
  wingRodNode1.childs += [wingRodNode]

  # Position the right back wing rod
  wingRodNode2 = sg.SceneGraphNode('wingRod2')
  wingRodNode2.transform = tr.matmul([tr.translate(-0.3, 1.65, 3)])
  wingRodNode2.childs += [wingRodNode]

  # Position the lift front wing rod
  wingRodNode3 = sg.SceneGraphNode('wingRod3')
  wingRodNode3.transform = tr.matmul([tr.translate(0.3, 1.65, -3)])
  wingRodNode3.childs += [wingRodNode]

  # Position the lift back wing rod
  wingRodNode4 = sg.SceneGraphNode('wingRod4')
  wingRodNode4.transform = tr.matmul([tr.translate(-0.3, 1.65, -3)])
  wingRodNode4.childs += [wingRodNode]

  # Postion the right front center wing rod
  wingRodNode5 = sg.SceneGraphNode('wingRod5')
  wingRodNode5.transform = tr.matmul(
      [tr.translate(0.3, 1.9, 1),
       tr.rotationX(pi / 3)])
  wingRodNode5.childs += [wingRodNode]

  # Postion the right back center wing rod
  wingRodNode6 = sg.SceneGraphNode('wingRod6')
  wingRodNode6.transform = tr.matmul(
      [tr.translate(-0.3, 1.9, 1),
       tr.rotationX(pi / 3)])
  wingRodNode6.childs += [wingRodNode]

  # Postion the left front center wing rod
  wingRodNode7 = sg.SceneGraphNode('wingRod7')
  wingRodNode7.transform = tr.matmul(
      [tr.translate(0.3, 1.9, -1),
       tr.rotationX(-pi / 3)])
  wingRodNode7.childs += [wingRodNode]

  # Postion the left back center wing rod
  wingRodNode8 = sg.SceneGraphNode('wingRod8')
  wingRodNode8.transform = tr.matmul(
      [tr.translate(-0.3, 1.9, -1),
       tr.rotationX(-pi / 3)])
  wingRodNode8.childs += [wingRodNode]

  # Group wing rod elements
  wingRodCompoundNode = sg.SceneGraphNode('wingRodCompund')
  wingRodCompoundNode.childs += [
      wingRodNode1, wingRodNode2, wingRodNode3, wingRodNode4, wingRodNode5,
      wingRodNode6, wingRodNode7, wingRodNode8
  ]

  # Group the wing elements
  wingCompoundNode = sg.SceneGraphNode('wings')
  wingCompoundNode.childs += [
      topWingNode, bottomWingNode, wingRodCompoundNode, rightWingCirleNode,
      leftWingCirleNode, rightTopWingRounderNode, rightBottomWingRounderNode,
      leftTopWingRounderNode, leftBottomWingRounderNode
  ]

  # Flaps
  # Create and position the horizontal flap
  horizontalFlapNode = sg.SceneGraphNode('horizontalFlap')
  horizontalFlapNode.transform = tr.matmul([
      tr.translate(-5.5, 1.4, 0),
      tr.uniformScale(0.5),
      tr.scale(1, 0.1, 1),
      tr.rotationZ(-pi / 2)
  ])
  horizontalFlapNode.childs += [cone]

  # Create and positio the vertical flap
  verticalFlapNode = sg.SceneGraphNode('verticalFlap')
  verticalFlapNode.transform = tr.matmul([
      tr.translate(-5.5, 1.4, 0),
      tr.uniformScale(0.5),
      tr.scale(1, 1, 0.1),
      tr.rotationZ(-pi / 2)
  ])
  verticalFlapNode.childs += [cone]

  # Group the flap elements
  flapsNode = sg.SceneGraphNode('flaps')
  flapsNode.childs += [horizontalFlapNode, verticalFlapNode]

  # Body
  # Create and position the center body
  bodyCenterNode = sg.SceneGraphNode('bodyCenter')
  bodyCenterNode.transform = tr.matmul(
      [tr.translate(-2, 1.4, 0),
       tr.scale(4, 0.7, 0.7),
       tr.rotationZ(pi / 2)])
  bodyCenterNode.childs += [cone]

  # Create and position the helix
  helixNode = sg.SceneGraphNode('helixOuter')
  helixNode.transform = tr.matmul([
      tr.translate(2.1, 1.4, 0),
      tr.scale(0.02, 0.02, 1),
      tr.rotationZ(pi / 2)
  ])
  helixNode.childs += [sphere]

  # Create and position the helix support
  helixSupportNode = sg.SceneGraphNode('helixSupport')
  helixSupportNode.transform = tr.matmul([
      tr.translate(2.05, 1.4, 0),
      tr.scale(0.1, 0.05, 0.05),
      tr.rotationZ(pi / 2)
  ])
  helixSupportNode.childs += [cylinder]

  # Create and position the seat
  seatNode = sg.SceneGraphNode('seat')
  seatNode.transform = tr.matmul([
      tr.translate(-0.4, 2, 0),
      tr.scale(0.01, 0.17, 0.17),
      tr.rotationZ(pi / 2)
  ])
  seatNode.childs += [cylinder]

  # Group the body elements
  bodyNode = sg.SceneGraphNode('body')
  bodyNode.childs += [bodyCenterNode, helixNode, helixSupportNode, seatNode]

  scene = sg.SceneGraphNode('system')
  scene.transform = tr.matmul([tr.translate(0, -1.5, 0)])
  scene.childs += [landingGearNode]
  scene.childs += [wingCompoundNode]
  scene.childs += [bodyNode]
  scene.childs += [flapsNode]
  return scene


if __name__ == "__main__":

  # Initialize glfw
  if not glfw.init():
    glfw.set_window_should_close(window, True)

  width = 800
  height = 800
  title = "Tarea 2"
  window = glfw.create_window(width, height, title, None, None)

  if not window:
    glfw.terminate()
    glfw.set_window_should_close(window, True)

  glfw.make_context_current(window)

  # Connecting the callback function 'on_key' to handle keyboard events
  glfw.set_key_callback(window, on_key)

  # Assembling the shader program (pipeline) with both shaders
  mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
  pipeline = ls.SimpleGouraudShaderProgram()

  # Telling OpenGL to use our shader program
  glUseProgram(mvpPipeline.shaderProgram)

  # Setting up the clear screen color
  glClearColor(0.85, 0.85, 0.85, 1.0)

  # As we work in 3D, we need to check which part is in front,
  # and which one is at the back
  glEnable(GL_DEPTH_TEST)

  # Creating shapes on GPU memory
  cpuAxis = bs.createAxis(7)
  gpuAxis = es.GPUShape().initBuffers()
  mvpPipeline.setupVAO(gpuAxis)
  gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

  #NOTA: Aqui creas un objeto con tu escena
  dibujo = createScene(pipeline)

  setPlot(pipeline, mvpPipeline)

  perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

  # glfw will swap buffers as soon as possible
  glfw.swap_interval(0)

  while not glfw.window_should_close(window):

    # Measuring performance
    perfMonitor.update(glfw.get_time())
    glfw.set_window_title(window, title + str(perfMonitor))

    # Using GLFW to check for input events
    glfw.poll_events()

    # Clearing the screen in both, color and depth
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Filling or not the shapes depending on the controller state
    if (controller.fillPolygon):
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    setView(pipeline, mvpPipeline)

    if controller.showAxis:
      glUseProgram(mvpPipeline.shaderProgram)
      glUniformMatrix4fv(
          glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE,
          tr.identity())
      mvpPipeline.drawCall(gpuAxis, GL_LINES)

    #NOTA: Aquí dibujas tu objeto de escena
    glUseProgram(pipeline.shaderProgram)
    sg.drawSceneGraphNode(dibujo, pipeline, "model")

    # Once the render is done, buffers are swapped, showing only the complete scene.
    glfw.swap_buffers(window)

  # freeing GPU memory
  gpuAxis.clear()
  dibujo.clear()

  glfw.terminate()
