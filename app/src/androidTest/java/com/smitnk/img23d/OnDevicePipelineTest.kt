package com.smitnk.img23d

import android.graphics.BitmapFactory
import android.os.Debug
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.smitnk.img23d.core.GlbReader
import com.smitnk.img23d.core.TripoSR
import com.smitnk.img23d.data.ModelManager
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Acceptance test on the real phone: original TripoSR weights -> on-device inference -> mesh -> GLB.
 * Needs the model installed (via the app or adb push). Results are printed to logcat tag "Img23DBench".
 *   ./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.smitnk.img23d.OnDevicePipelineTest
 */
@RunWith(AndroidJUnit4::class)
class OnDevicePipelineTest {
    @Test fun chairProducesRealGeometry() {
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val models = ModelManager(target)
        assumeTrue("TripoSR model not installed in ${models.dir}", models.manifest() != null)

        val bmp = InstrumentationRegistry.getInstrumentation().context.assets.open("chair.png").use { BitmapFactory.decodeStream(it) }
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h).also { bmp.getPixels(it, 0, w, 0, 0, w, h) }
        val rgba = IntArray(w * h * 4)
        for (i in px.indices) {
            rgba[i * 4] = (px[i] shr 16) and 255; rgba[i * 4 + 1] = (px[i] shr 8) and 255
            rgba[i * 4 + 2] = px[i] and 255; rgba[i * 4 + 3] = if (bmp.hasAlpha()) px[i] ushr 24 else 255
        }
        var peak = 0L
        val res = TripoSR(models.dir).run(rgba, w, h, TripoSR.Options().apply { resolution = 256; threads = 6 }) { _, _ ->
            peak = maxOf(peak, Debug.getPss())
        }
        val glb = File(target.cacheDir, "chair.glb").apply { outputStream().use { res.mesh.writeGlb(it) } }
        val back = GlbReader(glb.readBytes())
        val p = res.mesh.positions
        val ext = (0..2).map { a -> p.filterIndexed { i, _ -> i % 3 == a }.let { it.max() - it.min() } }
        Log.i("Img23DBench", "faces=${res.mesh.faceCount()} verts=${res.mesh.vertexCount()} peakPssMB=${peak / 1024} extent=$ext timingsMs=${res.timingsMs} glbBytes=${glb.length()}")
        assertTrue("too few faces", res.mesh.faceCount() > 10_000)
        assertTrue("degenerate extent $ext", ext.all { it in 0.2f..1.8f })
        assertTrue(back.indices.size == res.mesh.indices.size)
    }
}
