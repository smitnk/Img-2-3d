package com.smitnk.img23d.blender

import android.os.Process
import java.io.File

/** Resident memory of every process this app owns (the app plus proot + Blender children), in MB. */
object MemoryProbe {
    fun appTreeRssMb(): Int {
        val uid = Process.myUid().toString()
        var kb = 0L
        File("/proc").listFiles()?.forEach { d ->
            if (d.name.firstOrNull()?.isDigit() != true) return@forEach
            val status = runCatching { File(d, "status").readLines() }.getOrNull() ?: return@forEach
            if (status.firstOrNull { it.startsWith("Uid:") }?.split(Regex("\\s+"))?.getOrNull(1) != uid) return@forEach
            status.firstOrNull { it.startsWith("VmRSS:") }?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull()?.let { kb += it }
        }
        return (kb / 1024).toInt()
    }
}
