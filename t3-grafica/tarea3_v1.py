# coding=utf-8
"""Tarea 3"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath
from operator import add

__author__ = "Gabriel Rojas"
__license__ = "MIT"


# A class to store the application control
class Controller:

  def __init__(self):
    self.fillPolygon = True
    self.showAxis = True
    self.carPos = np.array([2.0, -0.037409, 5.0])
    self.carRotation = np.pi
    self.viewPos = self.carPos + np.array([0, 0.5, 2])
    self.at = self.carPos
    self.camUp = np.array([0, 1, 0])
    self.distance = 20

  # Create a method to move the car with the cam
  def moveCarWithCam(self, to):
    self.carPos += to

    self.viewPos = np.array([0, 0.5, -2])
    self.viewPos = self.viewPos @ tr.rotationY(-self.carRotation)[:3, :3]
    self.viewPos += self.carPos

    sg.findNode(car, "car1").transform = tr.matmul(
        [tr.translate(*controller.carPos),
         tr.rotationY(self.carRotation)])


controller = Controller()


def rotateByPoint(point, angle):
  print(tr.rotationAxis(angle, point, point + np.array([0, 1, 0])))


def setPlot(texPipeline, axisPipeline, lightPipeline):
  projection = tr.perspective(45, float(width) / float(height), 0.1, 100)

  glUseProgram(axisPipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1,
      GL_TRUE, projection)

  glUseProgram(texPipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE,
      projection)

  glUseProgram(lightPipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1,
      GL_TRUE, projection)

  glUniform3f(
      glGetUniformLocation(lightPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
  glUniform3f(
      glGetUniformLocation(lightPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
  glUniform3f(
      glGetUniformLocation(lightPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

  glUniform3f(
      glGetUniformLocation(lightPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
  glUniform3f(
      glGetUniformLocation(lightPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
  glUniform3f(
      glGetUniformLocation(lightPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

  glUniform3f(
      glGetUniformLocation(lightPipeline.shaderProgram, "lightPosition"), 5, 5,
      5)

  glUniform1ui(
      glGetUniformLocation(lightPipeline.shaderProgram, "shininess"), 1000)
  glUniform1f(
      glGetUniformLocation(lightPipeline.shaderProgram, "constantAttenuation"),
      0.1)
  glUniform1f(
      glGetUniformLocation(lightPipeline.shaderProgram, "linearAttenuation"),
      0.1)
  glUniform1f(
      glGetUniformLocation(lightPipeline.shaderProgram, "quadraticAttenuation"),
      0.01)


def setView(texPipeline, axisPipeline, lightPipeline):
  view = tr.lookAt(controller.viewPos, controller.at, controller.camUp)

  glUseProgram(axisPipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE,
      view)

  glUseProgram(texPipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

  glUseProgram(lightPipeline.shaderProgram)
  glUniformMatrix4fv(
      glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE,
      view)
  glUniform3f(
      glGetUniformLocation(lightPipeline.shaderProgram, "viewPosition"),
      controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])


def on_key(window, key, scancode, action, mods):

  if action != glfw.PRESS:
    return

  global controller

  if key == glfw.KEY_SPACE:
    controller.fillPolygon = not controller.fillPolygon

  elif key == glfw.KEY_LEFT_CONTROL:
    controller.showAxis = not controller.showAxis

  elif key == glfw.KEY_ESCAPE:
    glfw.set_window_should_close(window, True)


def createOFFShape(pipeline, filename, r, g, b):
  shape = readOFF(getAssetPath(filename), (r, g, b))
  gpuShape = es.GPUShape().initBuffers()
  pipeline.setupVAO(gpuShape)
  gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

  return gpuShape


def readOFF(filename, color):
  vertices = []
  normals = []
  faces = []

  with open(filename, 'r') as file:
    line = file.readline().strip()
    assert line == "OFF"

    line = file.readline().strip()
    aux = line.split(' ')

    numVertices = int(aux[0])
    numFaces = int(aux[1])

    for i in range(numVertices):
      aux = file.readline().strip().split(' ')
      vertices += [float(coord) for coord in aux[0:]]

    vertices = np.asarray(vertices)
    vertices = np.reshape(vertices, (numVertices, 3))
    print(f'Vertices shape: {vertices.shape}')

    normals = np.zeros((numVertices, 3), dtype=np.float32)
    print(f'Normals shape: {normals.shape}')

    for i in range(numFaces):
      aux = file.readline().strip().split(' ')
      aux = [int(index) for index in aux[0:]]
      faces += [aux[1:]]

      vecA = [
          vertices[aux[2]][0] - vertices[aux[1]][0],
          vertices[aux[2]][1] - vertices[aux[1]][1],
          vertices[aux[2]][2] - vertices[aux[1]][2]
      ]
      vecB = [
          vertices[aux[3]][0] - vertices[aux[2]][0],
          vertices[aux[3]][1] - vertices[aux[2]][1],
          vertices[aux[3]][2] - vertices[aux[2]][2]
      ]

      res = np.cross(vecA, vecB)
      normals[aux[1]][0] += res[0]
      normals[aux[1]][1] += res[1]
      normals[aux[1]][2] += res[2]

      normals[aux[2]][0] += res[0]
      normals[aux[2]][1] += res[1]
      normals[aux[2]][2] += res[2]

      normals[aux[3]][0] += res[0]
      normals[aux[3]][1] += res[1]
      normals[aux[3]][2] += res[2]
    #print(faces)
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, None]

    color = np.asarray(color)
    color = np.tile(color, (numVertices, 1))

    vertexData = np.concatenate((vertices, color), axis=1)
    vertexData = np.concatenate((vertexData, normals), axis=1)

    print(vertexData.shape)

    indices = []
    vertexDataF = []
    index = 0

    for face in faces:
      vertex = vertexData[face[0], :]
      vertexDataF += vertex.tolist()
      vertex = vertexData[face[1], :]
      vertexDataF += vertex.tolist()
      vertex = vertexData[face[2], :]
      vertexDataF += vertex.tolist()

      indices += [index, index + 1, index + 2]
      index += 3

    return bs.Shape(vertexDataF, indices)


def createGPUShape(pipeline, shape):
  gpuShape = es.GPUShape().initBuffers()
  pipeline.setupVAO(gpuShape)
  gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

  return gpuShape


def createTexturedArc(d):
  vertices = [d, 0.0, 0.0, 0.0, 0.0, d + 1.0, 0.0, 0.0, 1.0, 0.0]

  currentIndex1 = 0
  currentIndex2 = 1

  indices = []

  cont = 1
  cont2 = 1

  for angle in range(4, 185, 5):
    angle = np.radians(angle)
    rot = tr.rotationY(angle)
    p1 = rot.dot(np.array([[d], [0], [0], [1]]))
    p2 = rot.dot(np.array([[d + 1], [0], [0], [1]]))

    p1 = np.squeeze(p1)
    p2 = np.squeeze(p2)

    vertices.extend([p2[0], p2[1], p2[2], 1.0, cont / 4])
    vertices.extend([p1[0], p1[1], p1[2], 0.0, cont / 4])

    indices.extend([currentIndex1, currentIndex2, currentIndex2 + 1])
    indices.extend([currentIndex2 + 1, currentIndex2 + 2, currentIndex1])

    if cont > 4:
      cont = 0

    vertices.extend([p1[0], p1[1], p1[2], 0.0, cont / 4])
    vertices.extend([p2[0], p2[1], p2[2], 1.0, cont / 4])

    currentIndex1 = currentIndex1 + 4
    currentIndex2 = currentIndex2 + 4
    cont2 = cont2 + 1
    cont = cont + 1

  return bs.Shape(vertices, indices)


def createTiledFloor(dim):
  vert = np.array([[-0.5, 0.5, 0.5, -0.5], [-0.5, -0.5, 0.5, 0.5],
                   [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], np.float32)
  rot = tr.rotationX(-np.pi / 2)
  vert = rot.dot(vert)

  indices = [0, 1, 2, 2, 3, 0]

  vertFinal = []
  indexFinal = []
  cont = 0

  for i in range(-dim, dim, 1):
    for j in range(-dim, dim, 1):
      tra = tr.translate(i, 0.0, j)
      newVert = tra.dot(vert)

      v = newVert[:, 0][:-1]
      vertFinal.extend([v[0], v[1], v[2], 0, 1])
      v = newVert[:, 1][:-1]
      vertFinal.extend([v[0], v[1], v[2], 1, 1])
      v = newVert[:, 2][:-1]
      vertFinal.extend([v[0], v[1], v[2], 1, 0])
      v = newVert[:, 3][:-1]
      vertFinal.extend([v[0], v[1], v[2], 0, 0])

      ind = [elem + cont for elem in indices]
      indexFinal.extend(ind)
      cont = cont + 4

  return bs.Shape(vertFinal, indexFinal)


# TAREA3: Implementa la función "createHouse" que crea un objeto que representa una casa
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)
def createHouse(pipeline):
  # Creta a wall and a roof to position and transform to create a house
  wall = createWall(pipeline, "wall4.jpg")
  roof = createWall(pipeline, "roof1.jpg")

  wall1 = sg.SceneGraphNode("wall1")
  wall1.transform = tr.translate(0, 0, 0.5)
  wall1.childs += [wall]

  wall2 = sg.SceneGraphNode("wall2")
  wall2.transform = tr.translate(0, 0, -0.5)
  wall2.childs += [wall]

  wall3 = sg.SceneGraphNode("wall3")
  wall3.transform = tr.matmul(
      [tr.translate(0.5, 0, 0),
       tr.rotationY(np.pi / 2)])
  wall3.childs += [wall]

  wall4 = sg.SceneGraphNode("wall4")
  wall4.transform = tr.translate(-1, 0, 0)
  wall4.childs += [wall3]

  wall5 = sg.SceneGraphNode("wall5")
  wall5.transform = tr.matmul([
      tr.translate(0.4999, 0.75, -0.25),
      tr.rotationY(np.pi / 2),
      tr.rotationZ(np.pi / 4),
      tr.uniformScale(1 / np.sqrt(2))
  ])
  wall5.childs += [wall]

  wall6 = sg.SceneGraphNode("wall6")
  wall6.transform = tr.translate(-0.9998, 0, 0)
  wall6.childs += [wall5]

  roof1 = sg.SceneGraphNode("roof1")
  roof1.transform = tr.matmul(
      [tr.translate(0, 0.79, 0.71),
       tr.rotationX(-np.pi / 4)])
  roof1.childs += [roof]

  roof2 = sg.SceneGraphNode("roof2")
  roof2.transform = tr.rotationY(np.pi)
  roof2.childs += [roof1]

  scene = sg.SceneGraphNode("system")
  scene.childs += [wall1]
  scene.childs += [wall2]
  scene.childs += [wall3]
  scene.childs += [wall4]
  scene.childs += [wall5]
  scene.childs += [wall6]
  scene.childs += [roof1]
  scene.childs += [roof2]

  return scene


# TAREA3: Implementa la función "createWall" que crea un objeto que representa un muro
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)
def createWall(pipeline, asset):
  # Creates a wall with asset as it texture
  wallBaseShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
  wallBaseShape.texture = es.textureSimpleSetup(
      getAssetPath(asset), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR,
      GL_NEAREST)
  glGenerateMipmap(GL_TEXTURE_2D)

  wall = sg.SceneGraphNode("wall")
  wall.transform = tr.translate(0, 0.5, 0)
  wall.childs += [wallBaseShape]

  scene = sg.SceneGraphNode("system")
  scene.childs += [wall]

  return scene


# TAREA3: Esta función crea un grafo de escena especial para el auto.
def createCarScene(pipeline):
  chasis = createOFFShape(pipeline, 'alfa2.off', 1.0, 0.0, 0.0)
  wheel = createOFFShape(pipeline, 'wheel.off', 0.0, 0.0, 0.0)

  scale = 2.0
  rotatingWheelNode = sg.SceneGraphNode('rotatingWheel')
  rotatingWheelNode.childs += [wheel]

  chasisNode = sg.SceneGraphNode('chasis')
  chasisNode.transform = tr.uniformScale(scale)
  chasisNode.childs += [chasis]

  wheel1Node = sg.SceneGraphNode('wheel1')
  wheel1Node.transform = tr.matmul(
      [tr.uniformScale(scale),
       tr.translate(0.056390, 0.037409, 0.091705)])
  wheel1Node.childs += [rotatingWheelNode]

  wheel2Node = sg.SceneGraphNode('wheel2')
  wheel2Node.transform = tr.matmul(
      [tr.uniformScale(scale),
       tr.translate(-0.060390, 0.037409, -0.091705)])
  wheel2Node.childs += [rotatingWheelNode]

  wheel3Node = sg.SceneGraphNode('wheel3')
  wheel3Node.transform = tr.matmul(
      [tr.uniformScale(scale),
       tr.translate(-0.056390, 0.037409, 0.091705)])
  wheel3Node.childs += [rotatingWheelNode]

  wheel4Node = sg.SceneGraphNode('wheel4')
  wheel4Node.transform = tr.matmul(
      [tr.uniformScale(scale),
       tr.translate(0.066090, 0.037409, -0.091705)])
  wheel4Node.childs += [rotatingWheelNode]

  car1 = sg.SceneGraphNode('car1')
  car1.transform = tr.matmul(
      [tr.translate(*controller.carPos),
       tr.rotationY(controller.carRotation)])
  car1.childs += [chasisNode]
  car1.childs += [wheel1Node]
  car1.childs += [wheel2Node]
  car1.childs += [wheel3Node]
  car1.childs += [wheel4Node]

  scene = sg.SceneGraphNode('system')
  scene.childs += [car1]

  return scene


# TAREA3: Esta función crea toda la escena estática y texturada de esta aplicación.
# Por ahora ya están implementadas: la pista y el terreno
# En esta función debes incorporar las casas y muros alrededor de la pista
def createStaticScene(pipeline):

  roadBaseShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
  roadBaseShape.texture = es.textureSimpleSetup(
      getAssetPath("Road_001_basecolor.jpg"), GL_REPEAT, GL_REPEAT,
      GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
  glGenerateMipmap(GL_TEXTURE_2D)

  sandBaseShape = createGPUShape(pipeline, createTiledFloor(50))
  sandBaseShape.texture = es.textureSimpleSetup(
      getAssetPath("Sand 002_COLOR.jpg"), GL_REPEAT, GL_REPEAT,
      GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
  glGenerateMipmap(GL_TEXTURE_2D)

  arcShape = createGPUShape(pipeline, createTexturedArc(1.5))
  arcShape.texture = roadBaseShape.texture

  roadBaseNode = sg.SceneGraphNode('plane')
  roadBaseNode.transform = tr.rotationX(-np.pi / 2)
  roadBaseNode.childs += [roadBaseShape]

  arcNode = sg.SceneGraphNode('arc')
  arcNode.childs += [arcShape]

  sandNode = sg.SceneGraphNode('sand')
  sandNode.transform = tr.translate(0.0, -0.1, 0.0)
  sandNode.childs += [sandBaseShape]

  linearSector = sg.SceneGraphNode('linearSector')

  for i in range(10):
    node = sg.SceneGraphNode('road' + str(i) + '_ls')
    node.transform = tr.translate(0.0, 0.0, -1.0 * i)
    node.childs += [roadBaseNode]
    linearSector.childs += [node]

  linearSectorLeft = sg.SceneGraphNode('lsLeft')
  linearSectorLeft.transform = tr.translate(-2.0, 0.0, 5.0)
  linearSectorLeft.childs += [linearSector]

  linearSectorRight = sg.SceneGraphNode('lsRight')
  linearSectorRight.transform = tr.translate(2.0, 0.0, 5.0)
  linearSectorRight.childs += [linearSector]

  arcTop = sg.SceneGraphNode('arcTop')
  arcTop.transform = tr.translate(0.0, 0.0, -4.5)
  arcTop.childs += [arcNode]

  arcBottom = sg.SceneGraphNode('arcBottom')
  arcBottom.transform = tr.matmul(
      [tr.translate(0.0, 0.0, 5.5),
       tr.rotationY(np.pi)])
  arcBottom.childs += [arcNode]

  # Create the contention walls
  wall = createWall(pipeline, "wall3.jpg")
  contentionWalls = sg.SceneGraphNode("contentionWalls")

  contentionWall = sg.SceneGraphNode("contentionWall")
  contentionWall.transform = tr.matmul(
      [tr.translate(0, -0.8, 5),
       tr.rotationY(np.pi / 2),
       tr.scale(1, 1, 1)])
  contentionWall.childs += [wall]

  for i in range(10):
    node1 = sg.SceneGraphNode("innerContentionWall" + str(2 * i - 1))
    node1.transform = tr.translate(1.5, 0.0, -1.0 * i)
    node1.childs += [contentionWall]

    node2 = sg.SceneGraphNode("innerContentionWall" + str(2 * i))
    node2.transform = tr.translate(-1.5, 0.0, -1.0 * i)
    node2.childs += [contentionWall]

    node3 = sg.SceneGraphNode("outerContentionWall" + str(2 * i - 1))
    node3.transform = tr.translate(2.5, 0.0, -1.0 * i)
    node3.childs += [contentionWall]

    node4 = sg.SceneGraphNode("outerContentionWall" + str(2 * i))
    node4.transform = tr.translate(-2.5, 0.0, -1.0 * i)
    node4.childs += [contentionWall]

    contentionWalls.childs += [node1]
    contentionWalls.childs += [node2]
    contentionWalls.childs += [node3]
    contentionWalls.childs += [node4]

  # Create the houses
  house = createHouse(pipeline)
  houses = sg.SceneGraphNode("houses")

  houses_x = np.array([
      14.91456932, 18.67736079, -12.05143149, -14.82984088, 13.76227541,
      4.47342737, -8.04437192, -3.60394332, -9.96617537, -14.35432118,
      -5.55856333, -15.79221834, 19.58257251, -8.6018501, -13.46413808,
      3.96330519, 19.15333227, 3.59590668, 19.81539267, 10.07982558
  ])
  houses_z = np.array([
      6.7972175, -11.69465785, 4.56787729, 7.51469645, 14.96136501,
      -16.65371696, -13.2305607, -14.91431768, -15.05181169, 18.48694492,
      -14.08923421, -11.04175512, -6.20868106, 12.30586148, -14.56158148,
      9.93265521, 19.02352578, 14.96136501, -16.16128643, -9.73636059
  ])

  for i in range(20):
    node = sg.SceneGraphNode("house" + str(i + 1))
    node.transform = tr.translate(houses_x[i], 0, houses_z[i])
    node.childs += [house]
    houses.childs += [node]

  scene = sg.SceneGraphNode('system')
  scene.childs += [linearSectorLeft]
  scene.childs += [linearSectorRight]
  scene.childs += [arcTop]
  scene.childs += [arcBottom]
  scene.childs += [sandNode]
  scene.childs += [contentionWalls]
  scene.childs += [houses]

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
  axisPipeline = es.SimpleModelViewProjectionShaderProgram()
  texPipeline = es.SimpleTextureModelViewProjectionShaderProgram()
  lightPipeline = ls.SimpleGouraudShaderProgram()

  # Telling OpenGL to use our shader program
  glUseProgram(axisPipeline.shaderProgram)

  # Setting up the clear screen color
  glClearColor(0.85, 0.85, 0.85, 1.0)

  # As we work in 3D, we need to check which part is in front,
  # and which one is at the back
  glEnable(GL_DEPTH_TEST)

  # Creating shapes on GPU memory
  cpuAxis = bs.createAxis(7)
  gpuAxis = es.GPUShape().initBuffers()
  axisPipeline.setupVAO(gpuAxis)
  gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

  #NOTA: Aqui creas un objeto con tu escena
  dibujo = createStaticScene(texPipeline)
  car = createCarScene(lightPipeline)

  setPlot(texPipeline, axisPipeline, lightPipeline)

  perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

  # glfw will swap buffers as soon as possible
  glfw.swap_interval(0)

  moveRatio = 0.1

  while not glfw.window_should_close(window):

    # Measuring performance
    perfMonitor.update(glfw.get_time())
    glfw.set_window_title(window, title + str(perfMonitor))

    # Using GLFW to check for input events
    glfw.poll_events()

    # Moving car with WASD keys
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
      if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        controller.carRotation += np.pi / 100

      if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        controller.carRotation -= np.pi / 100

      to = np.array([
          moveRatio * np.sin(controller.carRotation), 0,
          moveRatio * np.cos(controller.carRotation)
      ])
      controller.moveCarWithCam(to)

    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
      if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        controller.carRotation += np.pi / 100

      if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        controller.carRotation -= np.pi / 100

      to = np.array([
          -moveRatio * np.sin(controller.carRotation), 0,
          -moveRatio * np.cos(controller.carRotation)
      ])
      controller.moveCarWithCam(to)

    # Clearing the screen in both, color and depth
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Filling or not the shapes depending on the controller state
    if (controller.fillPolygon):
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    setView(texPipeline, axisPipeline, lightPipeline)

    if controller.showAxis:
      glUseProgram(axisPipeline.shaderProgram)
      glUniformMatrix4fv(
          glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE,
          tr.identity())
      axisPipeline.drawCall(gpuAxis, GL_LINES)

    #NOTA: Aquí dibujas tu objeto de escena
    glUseProgram(texPipeline.shaderProgram)
    sg.drawSceneGraphNode(dibujo, texPipeline, "model")

    glUseProgram(lightPipeline.shaderProgram)
    sg.drawSceneGraphNode(car, lightPipeline, "model")

    # Once the render is done, buffers are swapped, showing only the complete scene.
    glfw.swap_buffers(window)

  # freeing GPU memory
  gpuAxis.clear()
  dibujo.clear()

  glfw.terminate()
