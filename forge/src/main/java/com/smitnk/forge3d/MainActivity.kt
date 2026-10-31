package com.smitnk.forge3d

import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.smitnk.forge3d.engine.Creation
import com.smitnk.forge3d.ui.ForgeActions
import com.smitnk.forge3d.ui.ForgeApp
import com.smitnk.forge3d.ui.ForgeTheme

class MainActivity : ComponentActivity() {
    private val vm: ForgeViewModel by viewModels()
    private var pendingExport: Creation? = null

    private val exporter = registerForActivityResult(ActivityResultContracts.CreateDocument("model/gltf-binary")) { uri ->
        val c = pendingExport
        pendingExport = null
        if (uri != null && c != null) contentResolver.openOutputStream(uri)?.let { vm.export(c, it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() { if (!vm.back()) finish() }
        })
        val actions = ForgeActions(
            export = { c -> pendingExport = c; exporter.launch("${c.title.replace(Regex("[^A-Za-z0-9_-]+"), "_")}.glb") },
            share = ::share,
        )
        setContent {
            val s by vm.state.collectAsStateWithLifecycle()
            LaunchedEffect(s.dest) {
                if (s.dest == Dest.Running) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
            ForgeTheme { ForgeApp(s, vm, actions) }
        }
    }

    private fun share(c: Creation) {
        val uri = FileProvider.getUriForFile(this, "$packageName.files", c.glb)
        val send = Intent(Intent.ACTION_SEND).setType("model/gltf-binary")
            .putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, "Share character"))
    }
}
