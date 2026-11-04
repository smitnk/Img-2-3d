package com.smitnk.img23d.body

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import com.smitnk.img23d.blender.BlenderRuntime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException

/**
 * Copies installed packs to / from a folder the user picked (Storage Access Framework tree), so a reinstall doesn't
 * mean downloading several GB again. Layout: <folder>/<pack dir name>/<files>, the same names the release uses.
 */
object PackFolder {
    private data class Child(val id: String, val name: String, val dir: Boolean, val size: Long)

    private fun children(ctx: Context, tree: Uri, parentId: String): List<Child> {
        val uri = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId)
        val cols = arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID, DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE, DocumentsContract.Document.COLUMN_SIZE)
        return ctx.contentResolver.query(uri, cols, null, null, null)?.use { c ->
            buildList {
                while (c.moveToNext()) add(Child(c.getString(0), c.getString(1),
                    c.getString(2) == DocumentsContract.Document.MIME_TYPE_DIR, c.getLong(3)))
            }
        } ?: emptyList()
    }

    private fun subfolder(ctx: Context, tree: Uri, name: String, create: Boolean): String? {
        val root = DocumentsContract.getTreeDocumentId(tree)
        children(ctx, tree, root).firstOrNull { it.dir && it.name == name }?.let { return it.id }
        if (!create) return null
        val made = DocumentsContract.createDocument(ctx.contentResolver, DocumentsContract.buildDocumentUriUsingTree(tree, root),
            DocumentsContract.Document.MIME_TYPE_DIR, name) ?: throw IOException("Cannot create folder $name")
        return DocumentsContract.getDocumentId(made)
    }

    /** Copy every file of an installed [pack] into <tree>/<pack dir name>/ (existing same-size files are kept). */
    suspend fun backup(ctx: Context, tree: Uri, pack: AssetPack, progress: (String) -> Unit) = withContext(Dispatchers.IO) {
        if (!pack.isReady()) return@withContext
        val folder = subfolder(ctx, tree, pack.dir.name, true)!!
        val have = children(ctx, tree, folder).associateBy { it.name }
        for (f in pack.dir.listFiles().orEmpty().filter { it.isFile && !it.name.endsWith(".part") }) {
            if (have[f.name]?.size == f.length()) continue
            progress("Backing up ${pack.dir.name}/${f.name}")
            have[f.name]?.let { DocumentsContract.deleteDocument(ctx.contentResolver, DocumentsContract.buildDocumentUriUsingTree(tree, it.id)) }
            val doc = DocumentsContract.createDocument(ctx.contentResolver, DocumentsContract.buildDocumentUriUsingTree(tree, folder),
                "application/octet-stream", f.name) ?: throw IOException("Cannot create ${f.name}")
            ctx.contentResolver.openOutputStream(doc)!!.use { out -> f.inputStream().use { it.copyTo(out, 1 shl 20) } }
        }
    }

    /**
     * Copy a pack back from <tree>/<pack dir name>/ and verify it against its own info file (bytes + sha256), then mark
     * it ready. Returns false when the folder has no copy of this pack.
     */
    suspend fun restore(ctx: Context, tree: Uri, pack: AssetPack, progress: (String) -> Unit): Boolean =
        withContext(Dispatchers.IO) {
            val folder = subfolder(ctx, tree, pack.dir.name, false) ?: return@withContext false
            val kids = children(ctx, tree, folder).associateBy { it.name }
            val infoName = pack.infoName
            val info = kids[infoName] ?: return@withContext false
            fun copy(c: Child, to: File) {
                ctx.contentResolver.openInputStream(DocumentsContract.buildDocumentUriUsingTree(tree, c.id))!!
                    .use { inp -> to.outputStream().use { inp.copyTo(it, 1 shl 20) } }
            }
            copy(info, File(pack.dir, infoName))
            val files = JSONObject(File(pack.dir, infoName).readText()).getJSONObject("files")
            for (name in files.keys()) {
                require(name.matches(Regex("[A-Za-z0-9_.-]+"))) { "Bad file name in pack info: $name" } // trust boundary
                val meta = files.getJSONObject(name)
                val target = File(pack.dir, name)
                if (target.length() != meta.getLong("bytes") || BlenderRuntime.sha256(target) != meta.getString("sha256")) {
                    val c = kids[name] ?: throw IOException("Backup of ${pack.dir.name} is missing $name")
                    progress("Restoring ${pack.dir.name}/$name")
                    copy(c, target)
                    if (BlenderRuntime.sha256(target) != meta.getString("sha256")) {
                        target.delete()
                        throw IOException("$name in the backup is damaged; download the pack instead")
                    }
                }
            }
            File(pack.dir, ".ready").writeText(File(pack.dir, infoName).readText())
            true
        }
}
