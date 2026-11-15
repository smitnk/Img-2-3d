package com.smitnk.img23d.body

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.net.InetAddress
import java.net.ServerSocket
import kotlin.concurrent.thread
import java.nio.file.Files

class AssetPackTest {
    private fun sha(b: ByteArray) = java.security.MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

    /** Tiny HTTP/1.0 server (the Android unit-test classpath has no com.sun.net.httpserver). */
    private class Server(val files: Map<String, ByteArray>, val info: String) {
        val socket = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))
        val port get() = socket.localPort
        init {
            thread(isDaemon = true) {
                while (!socket.isClosed) {
                    val c = try { socket.accept() } catch (e: Exception) { break }
                    c.use {
                        val line = it.getInputStream().bufferedReader().readLine() ?: return@use
                        val name = line.split(" ")[1].removePrefix("/")
                        val body = if (name == "p_info.json") info.toByteArray() else files[name]
                        val out = it.getOutputStream()
                        if (body == null) out.write("HTTP/1.0 404 Not Found\r\n\r\n".toByteArray())
                        else { out.write("HTTP/1.0 200 OK\r\nContent-Length: ${body.size}\r\n\r\n".toByteArray()); out.write(body) }
                        out.flush()
                    }
                }
            }
        }
        fun stop() = socket.close()
    }

    private fun serve(files: Map<String, ByteArray>, info: String) = Server(files, info)

    private fun pack(dir: File, server: Server) =
        AssetPack(dir, "http://127.0.0.1:${server.port}/", "p_info.json", 2, "a.bin")

    @Test fun failedUpdateKeepsOldPackReady() = runBlocking {
        val dir = Files.createTempDirectory("pack").toFile()
        val old = """{"version":2,"files":{"a.bin":{"bytes":3,"sha256":"${sha("old".toByteArray())}"}}}"""
        File(dir, "a.bin").writeText("old"); File(dir, ".ready").writeText(old)
        // the new pack serves wrong bytes -> checksum mismatch
        val bad = """{"version":3,"files":{"a.bin":{"bytes":3,"sha256":"${sha("new".toByteArray())}"}}}"""
        val server = serve(mapOf("a.bin" to "xxx".toByteArray()), bad)
        try {
            val p = pack(dir, server)
            assertTrue(p.isReady())
            runCatching { p.install { } }.let { assertTrue(it.isFailure) }
            assertTrue(p.isReady())
            assertEquals("old", File(dir, "a.bin").readText())
        } finally { server.stop() }
    }

    @Test fun goodUpdateReplacesFilesAndWritesReadyLast() = runBlocking {
        val dir = Files.createTempDirectory("pack").toFile()
        File(dir, "a.bin").writeText("old"); File(dir, "stale.bin").writeText("s")
        val new = """{"version":3,"files":{"a.bin":{"bytes":3,"sha256":"${sha("new".toByteArray())}"}}}"""
        val server = serve(mapOf("a.bin" to "new".toByteArray()), new)
        try {
            val p = pack(dir, server)
            assertFalse(p.isReady())
            p.install { }
            assertTrue(p.isReady())
            assertEquals("new", File(dir, "a.bin").readText())
            assertFalse(File(dir, "stale.bin").exists())
            assertFalse(File(dir, ".staging").exists())
        } finally { server.stop() }
    }
}
