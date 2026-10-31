package com.smitnk.img23d

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.smitnk.img23d.data.Model
import com.smitnk.img23d.ui.AppCallbacks
import com.smitnk.img23d.ui.AppRoot
import com.smitnk.img23d.ui.StudioTheme

class MainActivity : ComponentActivity() {
    private val vm: AppViewModel by viewModels()
    private var pendingExport: Pair<Model, Boolean>? = null
    private var pendingPhotoMode = PhotoMode.MESH

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { vm.pickImage(it, pendingPhotoMode) }
    }

    private val scriptPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let(vm::importScript) }

    private val exporter = registerForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
        val p = pendingExport
        pendingExport = null
        if (uri != null && p != null) contentResolver.openOutputStream(uri)?.let { vm.export(p.first, it, p.second) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() { if (!vm.back()) finish() }
        })
        val cb = AppCallbacks(
            pickImage = { mode ->
                pendingPhotoMode = mode
                picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            },
            pickScript = { scriptPicker.launch(arrayOf("text/x-python", "text/plain", "application/x-python-code", "application/octet-stream")) },
            export = { m, blend ->
                pendingExport = m to blend
                exporter.launch(m.name.replace(Regex("[^A-Za-z0-9_-]+"), "_") + if (blend) ".blend" else ".glb")
            },
            share = ::share,
            copy = { label, text ->
                getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText(label, text))
                Toast.makeText(this, "$label copied", Toast.LENGTH_SHORT).show()
            },
        )
        setContent {
            val state by vm.state.collectAsStateWithLifecycle()
            val settings by vm.settings.collectAsStateWithLifecycle()
            LaunchedEffect(state.dest, settings.keepScreenOn) {
                if (state.dest == Dest.Running && settings.keepScreenOn) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
            StudioTheme(settings.dynamicColor, settings.reduceMotion) { AppRoot(state, settings, vm, cb) }
        }
    }

    override fun onResume() {
        super.onResume()
        vm.refreshEngines()
    }

    private fun share(m: Model) {
        val uri = FileProvider.getUriForFile(this, "$packageName.files", m.glb)
        val send = Intent(Intent.ACTION_SEND).setType("model/gltf-binary")
            .putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, "Share 3D model"))
    }
}
