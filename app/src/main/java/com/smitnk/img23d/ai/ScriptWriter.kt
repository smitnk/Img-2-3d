package com.smitnk.img23d.ai

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Optional text-to-3D: Claude writes a Blender generator script, which then runs locally like any imported
 * script. Only the description (and, for repairs, the failing script + Blender log) leaves the phone.
 */
class ScriptWriter(private val contract: String, private val example: String) {

    data class Draft(val script: String, val notes: String)

    suspend fun write(apiKey: String, model: String, description: String, style: String, detail: String): Draft =
        call(apiKey, model, buildString {
            appendLine("<character>")
            appendLine(description.trim())
            appendLine("</character>")
            appendLine("<style>$style</style>")
            appendLine("<detail>$detail</detail>")
            append("Write the generator script for this character.")
        })

    suspend fun repair(apiKey: String, model: String, description: String, script: String, log: String): Draft =
        call(apiKey, model, buildString {
            appendLine("<character>${description.trim()}</character>")
            appendLine("<script>")
            appendLine(script)
            appendLine("</script>")
            appendLine("<blender_log>")
            appendLine(log.takeLast(4000))
            appendLine("</blender_log>")
            append("This script failed in Blender with the log above. Fix the cause and return the complete corrected script.")
        })

    private fun system() = """
You write Blender 4.2 Python generator scripts for a mobile app that runs them headless on a phone and exports
the result as a rigged GLB character. A failed script costs the user time, so correctness beats ambition.

<contract>
$contract
</contract>

<example_script>
$example
</example_script>

How to work:
- Decompose the character into named parts (head, torso, limbs, clothing, accessories) and build each with gm.part
  or bmesh, sized in metres for the requested proportions.
- Always build a T-pose and call gm.build_rig with the standard 17 bone names so the app's viewer and exports work.
- Use only APIs that exist in Blender 4.2 and work without a viewport. Prefer gm helpers and bmesh over bpy.ops.
- "detail" controls effort: low = few primitives; medium = shaped parts with bevels/subsurf; high = many parts,
  facial features, clothing layers, subsurf 2, still under ~200k triangles.

Reply with exactly two sections:
<notes>one or two sentences on what you built and any assumption</notes>
```python
<the complete script>
```
""".trimIndent()

    private suspend fun call(apiKey: String, model: String, user: String): Draft = withContext(Dispatchers.IO) {
        require(apiKey.isNotBlank()) { "Add your Anthropic API key in Settings → Text to 3D" }
        val body = JSONObject()
            .put("model", model)
            .put("max_tokens", 12000)
            .put("system", system())
            .put("messages", JSONArray().put(JSONObject().put("role", "user").put("content", user)))
        val conn = (URL("https://api.anthropic.com/v1/messages").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 20_000
            readTimeout = 180_000
            setRequestProperty("content-type", "application/json")
            setRequestProperty("x-api-key", apiKey.trim())
            setRequestProperty("anthropic-version", "2023-06-01")
        }
        try {
            conn.outputStream.use { it.write(body.toString().toByteArray()) }
            val code = conn.responseCode
            val text = (if (code in 200..299) conn.inputStream else conn.errorStream)?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                val msg = runCatching { JSONObject(text).getJSONObject("error").getString("message") }.getOrDefault(text.take(300))
                throw IOException("Claude API error $code: $msg")
            }
            val content = JSONObject(text).getJSONArray("content")
            val reply = (0 until content.length()).map { content.getJSONObject(it) }
                .filter { it.optString("type") == "text" }.joinToString("\n") { it.getString("text") }
            parse(reply)
        } finally {
            conn.disconnect()
        }
    }

    companion object {
        /** Extracts the python block and the notes; tolerates a reply that is only code. */
        fun parse(reply: String): Draft {
            val code = Regex("```(?:python|py)?\\s*\\n(.*?)```", RegexOption.DOT_MATCHES_ALL).find(reply)?.groupValues?.get(1)
                ?: reply.takeIf { "import bpy" in it || "generate_model" in it }
                ?: throw IOException("The model did not return a Python script")
            val notes = Regex("<notes>(.*?)</notes>", RegexOption.DOT_MATCHES_ALL).find(reply)?.groupValues?.get(1)?.trim().orEmpty()
            return Draft(code.trim() + "\n", notes)
        }
    }
}
