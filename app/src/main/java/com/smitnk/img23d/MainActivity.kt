package com.smitnk.img23d

import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
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
import com.smitnk.img23d.data.Project
import com.smitnk.img23d.ui.AppCallbacks
import com.smitnk.img23d.ui.AppRoot
import com.smitnk.img23d.ui.Img23DTheme

class MainActivity : ComponentActivity() {
    private val vm: AppViewModel by viewModels()
    private var pendingExport: Project? = null

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let(vm::pickImage)
    }

    private val exporter = registerForActivityResult(ActivityResultContracts.CreateDocument("model/gltf-binary")) { uri ->
        val p = pendingExport
        pendingExport = null
        if (uri != null && p != null) contentResolver.openOutputStream(uri)?.let { vm.exportGlb(p, it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (!vm.back()) finish()
            }
        })
        val cb = AppCallbacks(
            pickImage = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
            export = { p -> pendingExport = p; exporter.launch("img23d_${p.id}.glb") },
            share = ::share,
        )
        setContent {
            val state by vm.state.collectAsStateWithLifecycle()
            // Inference is foreground work; keep the screen on so the OS does not throttle or kill it.
            LaunchedEffect(state.screen) {
                if (state.screen == Screen.Generating) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
            Img23DTheme { AppRoot(state, vm, cb) }
        }
    }

    override fun onResume() {
        super.onResume()
        vm.refreshModels()
    }

    private fun share(p: Project) {
        val uri = FileProvider.getUriForFile(this, "$packageName.files", p.glb)
        val send = Intent(Intent.ACTION_SEND)
            .setType("model/gltf-binary")
            .putExtra(Intent.EXTRA_STREAM, uri)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, "Share 3D model"))
    }
}
