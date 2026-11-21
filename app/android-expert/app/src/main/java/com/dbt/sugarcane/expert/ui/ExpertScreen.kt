package com.dbt.sugarcane.expert.ui

import android.graphics.BitmapFactory
import android.util.Base64
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.dbt.sugarcane.expert.data.Classification
import com.dbt.sugarcane.expert.data.ExpertApi
import com.dbt.sugarcane.expert.data.ExpertResult
import com.dbt.sugarcane.expert.data.RedRotSeg
import com.dbt.sugarcane.expert.data.Settings
import com.dbt.sugarcane.expert.ui.theme.ChipGradient
import com.dbt.sugarcane.expert.ui.theme.HeaderGradient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream

private fun pct(x: Double): String = "%.1f%%".format(x * 100)

private fun parseColor(hex: String): Color = try {
    Color(android.graphics.Color.parseColor(hex))
} catch (_: Exception) {
    Color.Black
}

@Composable
fun ExpertScreen() {
    val context = LocalContext.current
    val settings = remember { Settings(context) }
    val api = remember { ExpertApi() }
    val scope = rememberCoroutineScope()

    var baseUrl by remember { mutableStateOf(settings.baseUrl) }
    var selectedBytes by remember { mutableStateOf<ByteArray?>(null) }
    var validate by remember { mutableStateOf(true) }
    var forceRedrot by remember { mutableStateOf(false) }
    var loading by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf<ExpertResult?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var history by remember { mutableStateOf(listOf<String>()) }
    var classes by remember { mutableStateOf(listOf<String>()) }

    fun setImage(bytes: ByteArray?) {
        selectedBytes = bytes
        result = null
        error = null
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            setImage(context.contentResolver.openInputStream(uri)?.use { it.readBytes() })
        }
    }
    val cameraLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bmp ->
        if (bmp != null) {
            val out = ByteArrayOutputStream()
            bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, out)
            setImage(out.toByteArray())
        }
    }

    fun loadSample() {
        setImage(context.assets.open("sample_redrot.jpg").use { it.readBytes() })
    }

    fun analyze() {
        val bytes = selectedBytes ?: return
        loading = true; error = null; result = null
        scope.launch {
            try {
                val r = withContext(Dispatchers.IO) {
                    api.predictExpert(baseUrl, bytes, "image.jpg", validate, forceRedrot)
                }
                result = r
                history = (listOf(r.overall_verdict.ifBlank { r.classification.predicted_class }) + history).take(10)
                if (classes.isEmpty()) {
                    classes = withContext(Dispatchers.IO) { api.fetchClasses(baseUrl) }
                }
            } catch (e: Exception) {
                error = e.message ?: "Request failed"
            } finally {
                loading = false
            }
        }
    }

    fun downloadReport() {
        val r = result ?: return
        scope.launch {
            try {
                val imgB64 = selectedBytes?.let { Base64.encodeToString(it, Base64.NO_WRAP) }
                val pdf = withContext(Dispatchers.IO) { api.downloadReport(baseUrl, r, imgB64) }
                // getExternalFilesDir can be null on some devices/emulators; fall back to internal storage.
                val dir = context.getExternalFilesDir(null) ?: context.filesDir
                val file = java.io.File(dir, "expert_report.pdf")
                withContext(Dispatchers.IO) { file.writeBytes(pdf) }
                Toast.makeText(context, "Report saved (${pdf.size} bytes): ${file.absolutePath}", Toast.LENGTH_LONG).show()
            } catch (e: Exception) {
                error = "Report failed: ${e.message}"
                Toast.makeText(context, "Report failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Box(
            Modifier.fillMaxWidth().clip(RoundedCornerShape(22.dp))
                .background(Brush.horizontalGradient(HeaderGradient)).padding(22.dp)
        ) {
            Column {
                Text("🌿 Sugarcane Disease — Expert View", style = MaterialTheme.typography.headlineSmall, color = Color.White, fontWeight = FontWeight.Bold)
                Text("Red Rot pilot · broad screen → Red Rot specialist", style = MaterialTheme.typography.bodySmall, color = Color(0xFFBBF7D0))
            }
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(baseUrl, { baseUrl = it }, label = { Text("API base URL") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                Button(onClick = { settings.baseUrl = baseUrl; baseUrl = settings.baseUrl }) { Text("Save") }
            }
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { galleryLauncher.launch("image/*") }) { Text("Gallery") }
                    OutlinedButton(onClick = { cameraLauncher.launch(null) }) { Text("Camera") }
                    OutlinedButton(onClick = { loadSample() }) { Text("Sample") }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(validate, { validate = it }); Text("Sugarcane validation gate")
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(forceRedrot, { forceRedrot = it }); Text("Force Red Rot specialist")
                }
                val preview = selectedBytes?.let { BitmapFactory.decodeByteArray(it, 0, it.size)?.asImageBitmap() }
                if (preview != null && result == null) {
                    Image(preview, "preview", Modifier.fillMaxWidth().height(220.dp), contentScale = ContentScale.Fit)
                }
                Button(onClick = { analyze() }, enabled = selectedBytes != null && !loading, modifier = Modifier.fillMaxWidth()) {
                    Text(if (loading) "Analyzing…" else "Analyze")
                }
                if (loading) LinearProgressIndicator(Modifier.fillMaxWidth())
            }
        }

        error?.let { Card(Modifier.fillMaxWidth()) { Text("⚠ $it", Modifier.padding(14.dp), color = MaterialTheme.colorScheme.error) } }

        result?.let { res ->
            VerdictCard(res)
            if (res.disagreement || res.low_confidence) CaptureGuidanceCard()
            Button(onClick = { downloadReport() }) { Text("⬇ Download PDF report") }
            ClassificationCard(res.classification)
            res.redrot_segmentation?.let { SegmentationCard(it, selectedBytes) }
            RemedialCard(res)
            ExpertCorrectionCard(res, classes) { pid, hash, pred, corr, notes ->
                scope.launch {
                    try {
                        withContext(Dispatchers.IO) { api.submitFeedback(baseUrl, pid, hash, pred, corr, notes) }
                        Toast.makeText(context, "Correction recorded for retraining", Toast.LENGTH_SHORT).show()
                    } catch (e: Exception) { error = e.message }
                }
            }
        }

        if (history.isNotEmpty()) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Text("History", fontWeight = FontWeight.Bold)
                    history.forEach { Text("· $it", style = MaterialTheme.typography.bodySmall) }
                }
            }
        }
    }
}

@Composable
private fun VerdictCard(res: ExpertResult) {
    val accent = when {
        res.disagreement -> Color(0xFFF59E0B)
        res.low_confidence -> Color(0xFF0EA5E9)
        else -> Color(0xFF10B981)
    }
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color.White)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("EXPERT VERDICT", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant, fontWeight = FontWeight.Bold)
            Text(res.overall_verdict.ifBlank { "—" }, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            if (res.disagreement) {
                Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Color(0xFFFEF3C7)).padding(8.dp)) {
                    Text("⚠ Models disagree — expert review suggested", color = Color(0xFF92400E), fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodySmall)
                }
            } else if (res.low_confidence) {
                Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Color(0xFFE0F2FE)).padding(8.dp)) {
                    Text("ⓘ Low screening confidence", color = Color(0xFF075985), fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodySmall)
                }
            }
            res.verdict_reason?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            Box(Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)).background(accent))
        }
    }
}

@Composable
private fun CaptureGuidanceCard() {
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFFECFEFF))) {
        Column(Modifier.padding(14.dp)) {
            Text("📸 Capture guidance", fontWeight = FontWeight.Bold)
            Text(
                "The whole-plant screen was unsure. Take one full-plant photo for screening and a close-up of the affected stalk/leaf section so the Red Rot specialist can confirm symptoms.",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun ClassificationCard(c: Classification) {
    var showDetail by remember { mutableStateOf(false) }
    val maxVote = (c.class_votes.values.maxOrNull() ?: 1).coerceAtLeast(1)
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Broad screen (whole-plant)", fontWeight = FontWeight.Bold)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.clip(RoundedCornerShape(999.dp)).background(Brush.horizontalGradient(ChipGradient)).padding(horizontal = 14.dp, vertical = 5.dp)) {
                    Text(c.predicted_class, style = MaterialTheme.typography.titleMedium, color = Color.White, fontWeight = FontWeight.Bold)
                }
                Spacer(Modifier.width(10.dp)); Text(pct(c.confidence))
            }
            LinearProgressIndicator(progress = { c.confidence.toFloat() }, modifier = Modifier.fillMaxWidth())
            Text("${c.agreeing_models}/${c.total_models} models agree (${pct(c.agreement_ratio)}) · ${c.voting_method} vote", style = MaterialTheme.typography.bodySmall)
            c.class_votes.entries.sortedByDescending { it.value }.forEach { (cls, n) ->
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(cls, Modifier.width(120.dp), style = MaterialTheme.typography.bodySmall, maxLines = 1)
                    Box(Modifier.weight(1f).height(10.dp).clip(RoundedCornerShape(999.dp)).background(Color(0xFFEEF5F2))) {
                        Box(
                            Modifier.fillMaxWidth(n.toFloat() / maxVote).height(10.dp).clip(RoundedCornerShape(999.dp))
                                .background(if (cls == c.predicted_class) Color(0xFF10B981) else Color(0xFFCBD5D0))
                        )
                    }
                    Text("$n", style = MaterialTheme.typography.bodySmall)
                }
            }
            TextButton(onClick = { showDetail = !showDetail }) { Text(if (showDetail) "Hide per-model detail" else "Show per-model detail") }
            if (showDetail) {
                HorizontalDivider()
                c.model_predictions.forEach { m ->
                    Text("${m.model} → ${m.prediction} (${pct(m.confidence)})", style = MaterialTheme.typography.bodySmall,
                        color = if (m.agrees_with_ensemble) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface)
                }
            }
        }
    }
}

@Composable
private fun SegmentationCard(seg: RedRotSeg, inputBytes: ByteArray?) {
    var opacity by remember { mutableStateOf(0.85f) }
    var showOverlay by remember { mutableStateOf(true) }
    val ar = if (seg.mask_width > 0 && seg.mask_height > 0) seg.mask_width.toFloat() / seg.mask_height else 1f
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Red Rot specialist (close-up) · ${seg.backbone}", fontWeight = FontWeight.Bold)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(showOverlay, { showOverlay = it }); Text("Overlay", style = MaterialTheme.typography.bodySmall)
                Spacer(Modifier.width(8.dp)); Text("Opacity", style = MaterialTheme.typography.bodySmall)
                Slider(opacity, { opacity = it }, modifier = Modifier.width(140.dp))
            }
            val base = inputBytes?.let { BitmapFactory.decodeByteArray(it, 0, it.size)?.asImageBitmap() }
            val overlay = remember(seg.overlay_png_b64) {
                if (seg.overlay_png_b64.isBlank()) null else {
                    val raw = Base64.decode(seg.overlay_png_b64, Base64.DEFAULT)
                    BitmapFactory.decodeByteArray(raw, 0, raw.size)?.asImageBitmap()
                }
            }
            Box(Modifier.fillMaxWidth().aspectRatio(ar).clip(RoundedCornerShape(12.dp))) {
                if (base != null) Image(base, "input", Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds)
                if (overlay != null && showOverlay) Image(overlay, "overlay", Modifier.fillMaxSize(), alpha = opacity, contentScale = ContentScale.FillBounds)
            }
            Text("Affected area: ${seg.affected_area_pct}%")
            Text("Dominant: ${seg.dominant_symptom ?: "—"}")
            Text("${seg.num_symptom_classes_detected} symptom class(es) detected", style = MaterialTheme.typography.bodySmall)
            seg.symptoms.forEach { s ->
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Box(Modifier.size(14.dp).clip(RoundedCornerShape(4.dp)).background(parseColor(s.color)))
                    Text(s.name, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
                    Text("${pct(s.pixel_fraction)} · ${pct(s.mean_confidence)}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

@Composable
private fun RemedialCard(res: ExpertResult) {
    val r = res.remedial ?: return
    if (r.summary == null) return
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Remedial measures" + if (r.status == "PLACEHOLDER") "  [PLACEHOLDER]" else "", fontWeight = FontWeight.Bold)
            Text(r.summary)
            r.immediate_actions?.takeIf { it.isNotEmpty() }?.let {
                Text("Immediate actions", fontWeight = FontWeight.SemiBold)
                it.forEach { a -> Text("· $a", style = MaterialTheme.typography.bodySmall) }
            }
            r.management?.takeIf { it.isNotEmpty() }?.let {
                Text("Management", fontWeight = FontWeight.SemiBold)
                it.forEach { a -> Text("· $a", style = MaterialTheme.typography.bodySmall) }
            }
        }
    }
}

@Composable
private fun ExpertCorrectionCard(
    res: ExpertResult,
    classes: List<String>,
    onSubmit: (String, String, String, String, String?) -> Unit,
) {
    var open by remember { mutableStateOf(res.disagreement) }
    var corrected by remember { mutableStateOf(res.classification.predicted_class) }
    var notes by remember { mutableStateOf("") }
    var menuOpen by remember { mutableStateOf(false) }
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color.White)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("🧑‍🔬 Expert correction" + if (res.disagreement) "  (review suggested)" else "", fontWeight = FontWeight.Bold)
            if (!open) {
                OutlinedButton(onClick = { open = true }) { Text("Correct this result") }
            } else {
                Box {
                    OutlinedButton(onClick = { menuOpen = true }) { Text("Disease: $corrected") }
                    DropdownMenu(menuOpen, { menuOpen = false }) {
                        (classes.ifEmpty { listOf(res.classification.predicted_class) }).forEach { c ->
                            DropdownMenuItem(text = { Text(c) }, onClick = { corrected = c; menuOpen = false })
                        }
                    }
                }
                OutlinedTextField(notes, { notes = it }, label = { Text("Notes (optional)") }, modifier = Modifier.fillMaxWidth())
                Button(onClick = {
                    onSubmit(res.classification.prediction_id, res.classification.image_hash, res.classification.predicted_class, corrected, notes.ifBlank { null })
                }) { Text("Submit correction") }
            }
        }
    }
}
