package com.smitnk.img23d.viewer

import android.annotation.SuppressLint
import android.content.Context
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import com.smitnk.img23d.core.GlbReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/** Orbit viewer for generated GLBs: 1 finger rotates, pinch zooms, 2 fingers pan, double-tap resets. */
@SuppressLint("ViewConstructor", "ClickableViewAccessibility")
class ModelView(context: Context) : GLSurfaceView(context) {
    private val renderer = MeshRenderer()
    private var lastX = 0f
    private var lastY = 0f
    private var lastTap = 0L
    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            renderer.distance = (renderer.distance / d.scaleFactor).coerceIn(0.6f, 8f)
            requestRender()
            return true
        }
    })

    init {
        setEGLContextClientVersion(3)
        setEGLConfigChooser(8, 8, 8, 8, 16, 4)
        setRenderer(renderer)
        renderMode = RENDERMODE_WHEN_DIRTY
        contentDescription = "Interactive 3D model viewer"
    }

    fun load(glb: ByteArray) {
        val r = GlbReader(glb)
        renderer.mesh = r
        resetCamera()
    }

    fun resetCamera() {
        renderer.yaw = 30f; renderer.pitch = 15f; renderer.distance = 2.6f; renderer.panX = 0f; renderer.panY = 0f
        requestRender()
    }

    override fun onTouchEvent(e: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(e)
        val x = (0 until e.pointerCount).map { e.getX(it) }.average().toFloat()
        val y = (0 until e.pointerCount).map { e.getY(it) }.average().toFloat()
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                val now = System.currentTimeMillis()
                if (now - lastTap < 300) resetCamera()
                lastTap = now
                lastX = x; lastY = y
            }
            MotionEvent.ACTION_POINTER_DOWN, MotionEvent.ACTION_POINTER_UP -> { lastX = x; lastY = y }
            MotionEvent.ACTION_MOVE -> {
                val dx = x - lastX
                val dy = y - lastY
                if (e.pointerCount == 1) {
                    renderer.yaw += dx * 0.4f
                    renderer.pitch = (renderer.pitch + dy * 0.4f).coerceIn(-89f, 89f)
                } else if (!scaleDetector.isInProgress || e.pointerCount >= 2) {
                    val k = renderer.distance / height
                    renderer.panX += dx * k
                    renderer.panY -= dy * k
                }
                lastX = x; lastY = y
                requestRender()
            }
        }
        parent?.requestDisallowInterceptTouchEvent(true)
        return true
    }
}

private class MeshRenderer : GLSurfaceView.Renderer {
    @Volatile var yaw = 30f
    @Volatile var pitch = 15f
    @Volatile var distance = 2.6f
    @Volatile var panX = 0f
    @Volatile var panY = 0f

    // Uploaded lazily on the GL thread: data set before the EGL context exists (or after a context loss) must survive.
    @Volatile var mesh: GlbReader? = null
    private var uploaded: GlbReader? = null
    private var program = 0
    private var vao = 0
    private var indexCount = 0
    private val proj = FloatArray(16)
    private val view = FloatArray(16)
    private val model = FloatArray(16)
    private val mvp = FloatArray(16)
    private val mv = FloatArray(16)

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0.06f, 0.09f, 0.16f, 1f)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        program = link(VS, FS)
        vao = 0
        uploaded = null
    }

    private fun upload(r: GlbReader) {
        if (vao != 0) GLES30.glDeleteVertexArrays(1, intArrayOf(vao), 0)
        val p = r.positions
        val lo = FloatArray(3) { Float.MAX_VALUE }
        val hi = FloatArray(3) { -Float.MAX_VALUE }
        for (i in p.indices) { lo[i % 3] = minOf(lo[i % 3], p[i]); hi[i % 3] = maxOf(hi[i % 3], p[i]) }
        val ext = (0..2).maxOf { hi[it] - lo[it] }.coerceAtLeast(1e-6f)
        Matrix.setIdentityM(model, 0)
        Matrix.scaleM(model, 0, 1.6f / ext, 1.6f / ext, 1.6f / ext)
        Matrix.translateM(model, 0, -(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -(lo[2] + hi[2]) / 2)
        val colors = r.colors ?: FloatArray(p.size) { 0.8f }

        val ids = IntArray(1)
        GLES30.glGenVertexArrays(1, ids, 0); vao = ids[0]
        GLES30.glBindVertexArray(vao)
        attrib(0, p); attrib(1, r.normals); attrib(2, colors)
        val ibo = IntArray(1)
        GLES30.glGenBuffers(1, ibo, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, ibo[0])
        val ib = ByteBuffer.allocateDirect(r.indices.size * 4).order(ByteOrder.nativeOrder()).asIntBuffer().put(r.indices).position(0)
        GLES30.glBufferData(GLES30.GL_ELEMENT_ARRAY_BUFFER, r.indices.size * 4, ib, GLES30.GL_STATIC_DRAW)
        GLES30.glBindVertexArray(0)
        indexCount = r.indices.size
    }

    private fun attrib(loc: Int, data: FloatArray) {
        val b = IntArray(1)
        GLES30.glGenBuffers(1, b, 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, b[0])
        val fb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(data).position(0)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, data.size * 4, fb, GLES30.GL_STATIC_DRAW)
        GLES30.glEnableVertexAttribArray(loc)
        GLES30.glVertexAttribPointer(loc, 3, GLES30.GL_FLOAT, false, 0, 0)
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        GLES30.glViewport(0, 0, w, h)
        Matrix.perspectiveM(proj, 0, 40f, w.toFloat() / h.coerceAtLeast(1), 0.05f, 50f)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        val m = mesh
        if (m != null && m !== uploaded) { upload(m); uploaded = m }
        if (uploaded == null) return
        val yr = Math.toRadians(yaw.toDouble())
        val pr = Math.toRadians(pitch.toDouble())
        val ex = (distance * Math.cos(pr) * Math.sin(yr)).toFloat()
        val ey = (distance * Math.sin(pr)).toFloat()
        val ez = (distance * Math.cos(pr) * Math.cos(yr)).toFloat()
        Matrix.setLookAtM(view, 0, ex, ey, ez, 0f, 0f, 0f, 0f, 1f, 0f)
        val pan = FloatArray(16).also { Matrix.setIdentityM(it, 0); Matrix.translateM(it, 0, panX, panY, 0f) }
        val v2 = FloatArray(16)
        Matrix.multiplyMM(v2, 0, pan, 0, view, 0)
        Matrix.multiplyMM(mv, 0, v2, 0, model, 0)
        Matrix.multiplyMM(mvp, 0, proj, 0, mv, 0)
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(program, "uMvp"), 1, false, mvp, 0)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(program, "uMv"), 1, false, mv, 0)
        GLES30.glBindVertexArray(vao)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, indexCount, GLES30.GL_UNSIGNED_INT, 0)
        GLES30.glBindVertexArray(0)
    }

    private fun link(vs: String, fs: String): Int {
        fun compile(type: Int, src: String) = GLES30.glCreateShader(type).also {
            GLES30.glShaderSource(it, src); GLES30.glCompileShader(it)
            val ok = IntArray(1); GLES30.glGetShaderiv(it, GLES30.GL_COMPILE_STATUS, ok, 0)
            check(ok[0] != 0) { GLES30.glGetShaderInfoLog(it) }
        }
        return GLES30.glCreateProgram().also {
            GLES30.glAttachShader(it, compile(GLES30.GL_VERTEX_SHADER, vs))
            GLES30.glAttachShader(it, compile(GLES30.GL_FRAGMENT_SHADER, fs))
            GLES30.glLinkProgram(it)
        }
    }

    companion object {
        private const val VS = """#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
uniform mat4 uMvp;
uniform mat4 uMv;
out vec3 vNormal;
out vec3 vColor;
void main() {
  // model matrix is uniform scale + translate, so mat3(uMv) is fine for normals after normalize
  vNormal = normalize(mat3(uMv) * aNormal);
  vColor = aColor;
  gl_Position = uMvp * vec4(aPos, 1.0);
}"""
        private const val FS = """#version 300 es
precision mediump float;
in vec3 vNormal;
in vec3 vColor;
out vec4 frag;
void main() {
  vec3 n = normalize(vNormal);
  float d = abs(n.z);           // two-sided headlight
  float rim = pow(1.0 - d, 3.0) * 0.15;
  vec3 c = vColor * (0.35 + 0.65 * d) + rim;
  frag = vec4(pow(c, vec3(1.0 / 1.1)), 1.0);
}"""
    }
}
