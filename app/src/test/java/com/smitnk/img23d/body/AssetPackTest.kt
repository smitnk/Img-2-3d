package com.smitnk.img23d.body

import com.sun.net.httpserver.HttpServer
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.net.InetSocketAddress
import java.nio.file.Files

class AssetPackTest {
    private fun sha(b: ByteArray) = java.security.MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

    private fun serve(files: Map<String, ByteArray>, info: String): HttpServer =
        HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            createContext("/") { ex ->
                val name = ex.requestURI.path.removePrefix("/")
                val body = if (name == "p_info.json") info.toByteArray() else files[name]
                if (body == null) ex.sendResponseHeaders(404, -1) else { ex.sendResponseHeaders(200, body.size.toLong()); ex.responseBody.use { it.write(body) } }
                ex.close()
            }
            start()
        }

    private fun pack(dir: File, server: HttpServer) =
        AssetPack(dir, "http://127.0.0.1:${server.address.port}/", "p_info.json", 2, "a.bin")

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
        } finally { server.stop(0) }
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
        } finally { server.stop(0) }
    }
}
