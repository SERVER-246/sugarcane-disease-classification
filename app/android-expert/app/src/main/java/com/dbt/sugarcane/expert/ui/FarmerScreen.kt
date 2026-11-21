package com.dbt.sugarcane.expert.ui

import android.graphics.BitmapFactory
import android.util.Base64
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.dbt.sugarcane.expert.data.ExpertApi
import com.dbt.sugarcane.expert.data.ExpertResult
import com.dbt.sugarcane.expert.data.Settings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream

private fun prettyDisease(name: String) = name.replace("_", " ")

@Composable
fun FarmerScreen(locale: String) {
    val context = LocalContext.current
    val settings = remember { Settings(context) }
    val api = remember { ExpertApi() }
    val scope = rememberCoroutineScope()

    var bytes by remember { mutableStateOf<ByteArray?>(null) }
    var loading by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf<ExpertResult?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var needsConfirm by remember { mutableStateOf(false) }

    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicturePreview()) { bmp ->
        if (bmp != null) {
            val out = ByteArrayOutputStream()
            bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, out)
            bytes = out.toByteArray(); result = null; error = null; needsConfirm = false
        }
    }
    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            result = null; error = null; needsConfirm = false
        }
    }

    fun analyze(validate: Boolean = true) {
        val b = bytes ?: return
        loading = true; error = null; result = null; needsConfirm = false
        scope.launch {
            try {
                val r = withContext(Dispatchers.IO) {
                    api.predictExpert(settings.baseUrl, b, "image.jpg", validate = validate, forceRedrot = false)
                }
                val failedValidation = validate &&
                    (r.overall_verdict == "Not a sugarcane image" || r.classification.validation?.passed == false)
                if (failedValidation) {
                    needsConfirm = true  // gentle fallback, don't hard-block
                } else {
                    result = r
                }
            } catch (e: Exception) {
                error = e.message?.let { if (it.contains("Unable to resolve host") || it.contains("Failed to connect")) Strings.t("server_unreachable", locale) else it }
            } finally {
                loading = false
            }
        }
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(Strings.t("farmer_tagline", locale), style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)

        val preview = bytes?.let { BitmapFactory.decodeByteArray(it, 0, it.size)?.asImageBitmap() }
        if (result == null) {
            if (preview != null) {
                Image(preview, "preview", Modifier.fillMaxWidth().height(220.dp).clip(RoundedCornerShape(16.dp)), contentScale = ContentScale.Fit)
            }
            Button(
                onClick = { cameraLauncher.launch(null) },
                modifier = Modifier.fillMaxWidth().height(64.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF059669)),
            ) { Text(Strings.t("capture", locale), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold) }
            OutlinedButton(onClick = { galleryLauncher.launch("image/*") }, modifier = Modifier.fillMaxWidth().height(56.dp)) {
                Text(Strings.t("gallery", locale), fontWeight = FontWeight.Bold)
            }
            if (bytes != null && !needsConfirm) {
                Button(
                    onClick = { analyze(true) }, enabled = !loading,
                    modifier = Modifier.fillMaxWidth().height(60.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0EA5E9)),
                ) { Text(if (loading) Strings.t("analyzing", locale) else Strings.t("analyze", locale), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold) }
            }
            if (needsConfirm) {
                Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFFFFFBEB))) {
                    Column(Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text("⚠ ${Strings.t("not_sugarcane", locale)}", color = Color(0xFF92400E), fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center)
                        Button(
                            onClick = { analyze(false) }, enabled = !loading,
                            modifier = Modifier.fillMaxWidth().height(56.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0EA5E9)),
                        ) { Text(if (loading) Strings.t("analyzing", locale) else Strings.t("analyze_anyway", locale), fontWeight = FontWeight.Bold) }
                    }
                }
            }
            error?.let {
                Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFFFEF2F2))) {
                    Text("⚠ $it", Modifier.padding(14.dp), color = Color(0xFFB91C1C), fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center)
                }
            }
        }

        result?.let { res ->
            val status = res.farmer_status
            val (c1, c2) = when (status) {
                "green" -> Color(0xFF10B981) to Color(0xFF059669)
                "red" -> Color(0xFFEF4444) to Color(0xFFB91C1C)
                else -> Color(0xFFF59E0B) to Color(0xFFD97706)
            }
            val icon = when (status) { "green" -> "✓"; "red" -> "!"; else -> "?" }
            val headline = when {
                status == "green" -> Strings.t("status_healthy", locale)
                status == "red" -> Strings.t("status_disease", locale, prettyDisease(res.farmer_disease ?: ""))
                res.farmer_disease == "Red_rot" -> Strings.t("status_possible_redrot", locale)
                else -> Strings.t("status_uncertain", locale)
            }

            Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(Brush.linearGradient(listOf(c1, c2))).padding(24.dp)) {
                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
                    Box(Modifier.size(64.dp).clip(CircleShape).background(Color.White.copy(alpha = 0.22f)), contentAlignment = Alignment.Center) {
                        Text(icon, style = MaterialTheme.typography.headlineMedium, color = Color.White, fontWeight = FontWeight.Bold)
                    }
                    Text(headline, style = MaterialTheme.typography.headlineSmall, color = Color.White, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center, modifier = Modifier.padding(top = 10.dp))
                    if (status == "green") {
                        Text("${Strings.t("confidence", locale)}: ${Strings.t("likely", locale)}", color = Color.White, modifier = Modifier.padding(top = 6.dp))
                    }
                }
            }

            val seg = res.redrot_segmentation
            val showOverlay = status == "red" && res.farmer_disease == "Red_rot" && seg != null && seg.affected_area_pct > 0
            if (showOverlay && seg != null) {
                val ar = if (seg.mask_width > 0 && seg.mask_height > 0) seg.mask_width.toFloat() / seg.mask_height else 1f
                val overlay = remember(seg.overlay_png_b64) {
                    if (seg.overlay_png_b64.isBlank()) null else {
                        val raw = Base64.decode(seg.overlay_png_b64, Base64.DEFAULT)
                        BitmapFactory.decodeByteArray(raw, 0, raw.size)?.asImageBitmap()
                    }
                }
                Box(Modifier.fillMaxWidth().aspectRatio(ar).clip(RoundedCornerShape(16.dp))) {
                    if (preview != null) Image(preview, "crop", Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds)
                    if (overlay != null) Image(overlay, "signs", Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds)
                }
                Text("${Strings.t("redrot_symptoms", locale)} · ${Strings.t("affected_area", locale)} ${seg.affected_area_pct}%", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else if (preview != null) {
                Image(preview, "crop", Modifier.fillMaxWidth().height(220.dp).clip(RoundedCornerShape(16.dp)), contentScale = ContentScale.Fit)
            }

            if (status == "amber") {
                Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFFECFEFF))) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(Strings.t("guidance_title", locale), fontWeight = FontWeight.Bold)
                        Text(Strings.t("guidance_body", locale), style = MaterialTheme.typography.bodyMedium)
                        Text(Strings.t("consult_expert", locale), fontWeight = FontWeight.Bold, color = Color(0xFF075985))
                    }
                }
            } else {
                val action = res.remedial?.immediate_actions?.firstOrNull() ?: res.remedial?.summary
                if (action != null) {
                    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color(0xFFF4FAF7))) {
                        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text(Strings.t("recommended_action", locale), fontWeight = FontWeight.Bold, color = Color(0xFF065F46))
                            Text(action)
                        }
                    }
                }
            }

            OutlinedButton(onClick = { bytes = null; result = null; error = null }, modifier = Modifier.fillMaxWidth().height(56.dp)) {
                Text(Strings.t("retake", locale), fontWeight = FontWeight.Bold)
            }
        }
    }
}
