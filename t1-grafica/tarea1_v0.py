import glfw
import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy
import sys

__author__ = "Gabriel Rojas"
__license__ = "MIT"

# We will use 32 bits data, so an integer has 4 bytes
# 1 byte = 8 bits
SIZE_IN_BYTES = 4

def crear_dama(x,y,r,g,b,radius):

    circle = []
    for angle in range(0,360,10):
        circle.extend([x, y, 0.0, r, g, b])
        circle.extend([x+numpy.cos(numpy.radians(angle))*radius, 
                       y+numpy.sin(numpy.radians(angle))*radius, 
                       0.0, r, g, b])
        circle.extend([x+numpy.cos(numpy.radians(angle+10))*radius, 
                       y+numpy.sin(numpy.radians(angle+10))*radius, 
                       0.0, r, g, b])
    
    return numpy.array(circle, dtype = numpy.float32)


def createQuad(dx, dy, color):
    # Devuelve las coordenadas y color de un cuadrado del tablero
    
    # Ya que estamos usando rgb expandimos color en un arreglo de largo 3
    color = [float(color)]*3 
    # Ahora creamos el arreglo con los vertices del cuadrado y su respectivo color
    vertexData = np.array([
         0.8-0.2*dx    , -0.8+0.2*dy    , 0.0, *color,
         0.8-0.2*(dx+1), -0.8+0.2*dy    , 0.0, *color,
         0.8-0.2*(dx+1), -0.8+0.2*(dy+1), 0.0, *color,
         0.8-0.2*(dx+1), -0.8+0.2*(dy+1), 0.0, *color,
         0.8-0.2*dx    , -0.8+0.2*(dy+1), 0.0, *color,
         0.8-0.2*dx    , -0.8+0.2*dy    , 0.0, *color
        ], dtype = np.float32)

    return vertexData


def crearDamas():
    # Devuelve las posiciones y colores de las damas

    # Creamos una lista con todas las damas y su respectivo color
    damas = list()
    for i in range(1,9):
        for j in range(1,9):
            if 4<=j<=5: # No queremos damas al medio del tablero
                continue
            if (i+j)%2==0: # Si en la casilla va una dama
                if j<4: # Azules
                    damas.extend(crear_dama(0.9-0.2*i,-0.9+0.2*j, 0.0, 0.0, 1.0, 0.07))
                elif j>5: # Rojas
                    damas.extend(crear_dama(0.9-0.2*i,-0.9+0.2*j, 1.0, 0.0, 0.0, 0.07))

    return np.array(damas, dtype=np.float32)


def crearTablero():
    # Devuelve el tablero 

    # Creamos una lista con todos los cuadrados del tablero
    tablero = list()
    for i in range(8):
        for j in range(8):
            tablero.extend(createQuad(i, j, (i+j)%2))

    return np.array(tablero, dtype=np.float32)


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Tarea 1", None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Defining shaders for our pipeline
    vertex_shader = """
    #version 330
    in vec3 position;
    in vec3 color;

    out vec3 newColor;
    void main()
    {
        gl_Position = vec4(position, 1.0f);
        newColor = color;
    }
    """

    fragment_shader = """
    #version 330
    in vec3 newColor;

    out vec4 outColor;
    void main()
    {
        outColor = vec4(newColor, 1.0f);
    }
    """

    # Binding artificial vertex array object for validation
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # Assembling the shader program (pipeline) with both shaders
    shaderProgram = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    # Creacion de vertices
    # Cada figura debe estar asociado a un Vertex Buffer Object (VBO)
    tablero = crearTablero()
    vboTablero = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vboTablero)
    glBufferData(GL_ARRAY_BUFFER, len(tablero) * SIZE_IN_BYTES, tablero, GL_STATIC_DRAW)

    # Cada figura debe estar asociado a un Vertex Buffer Object (VBO)
    damas = crearDamas()
    vboDama = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vboDama)
    glBufferData(GL_ARRAY_BUFFER, len(damas) * SIZE_IN_BYTES, damas, GL_STATIC_DRAW)

    # Telling OpenGL to use our shader program
    glUseProgram(shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.5,0.5, 0.5, 1.0)

    glClear(GL_COLOR_BUFFER_BIT)

    # Setup del tablero
    glBindBuffer(GL_ARRAY_BUFFER, vboTablero)
    position = glGetAttribLocation(shaderProgram, "position") 
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shaderProgram, "color") 
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)

    # Dibujamos el tablero
    glDrawArrays(GL_TRIANGLES, 0, int(len(tablero)/6))
    
    # Setup de las damas
    glBindBuffer(GL_ARRAY_BUFFER, vboDama)
    position = glGetAttribLocation(shaderProgram, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shaderProgram, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
    
    # Dibujamos las damas
    glDrawArrays(GL_TRIANGLES, 0, int(len(damas)/6))

    # Moving our draw to the active color buffer
    glfw.swap_buffers(window)

    # Waiting to close the window
    while not glfw.window_should_close(window):

        # Getting events from GLFW
        glfw.poll_events()
        
    glfw.terminate()
