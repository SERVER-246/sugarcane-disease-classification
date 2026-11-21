package com.dbt.sugarcane.expert

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.dbt.sugarcane.expert.data.Settings
import com.dbt.sugarcane.expert.ui.ExpertScreen
import com.dbt.sugarcane.expert.ui.FarmerScreen
import com.dbt.sugarcane.expert.ui.Strings
import com.dbt.sugarcane.expert.ui.theme.ExpertColors

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = ExpertColors) {
                Surface(color = ExpertColors.background) {
                    AppRoot()
                }
            }
        }
    }
}

@Composable
private fun AppRoot() {
    val context = LocalContext.current
    val settings = remember { Settings(context) }
    var view by remember { mutableStateOf(settings.view) }
    var locale by remember { mutableStateOf(settings.locale) }

    fun choose(v: String) { view = v; settings.view = v }
    fun switchLocale(l: String) { locale = l; settings.locale = l }

    if (view == null) {
        FirstRun(locale, onChoose = ::choose, onLocale = ::switchLocale)
        return
    }

    Column(Modifier.fillMaxSize()) {
        TopBar(view!!, locale, onView = ::choose, onLocale = ::switchLocale)
        if (view == "farmer") FarmerScreen(locale) else ExpertScreen()
    }
}

@Composable
private fun TopBar(view: String, locale: String, onView: (String) -> Unit, onLocale: (String) -> Unit) {
    Row(
        Modifier.fillMaxWidth().padding(10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        SegToggle("🌾 " + Strings.t("farmer_view", locale), view == "farmer") { onView("farmer") }
        SegToggle("🔬 " + Strings.t("expert_view", locale), view == "expert") { onView("expert") }
        Box(Modifier.weight(1f))
        Strings.LOCALES.forEach { (code, label) ->
            LangChip(label, locale == code) { onLocale(code) }
        }
    }
}

@Composable
private fun SegToggle(label: String, on: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (on) Color(0xFF059669) else Color(0xFFEEF5F2),
            contentColor = if (on) Color.White else Color(0xFF334E45),
        ),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp, vertical = 6.dp),
    ) { Text(label, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodySmall) }
}

@Composable
private fun LangChip(label: String, on: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (on) Color(0xFF065F46) else Color.White,
            contentColor = if (on) Color.White else Color(0xFF334E45),
        ),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp, vertical = 4.dp),
    ) { Text(label, style = MaterialTheme.typography.bodySmall) }
}

@Composable
private fun FirstRun(locale: String, onChoose: (String) -> Unit, onLocale: (String) -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(26.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(Strings.t("app_title", locale), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Text(Strings.t("choose_view", locale), style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 16.dp))
        Text(Strings.t("choose_view_desc", locale), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)

        Button(
            onClick = { onChoose("farmer") },
            modifier = Modifier.fillMaxWidth().height(64.dp).padding(top = 24.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF059669)),
        ) { Text("🌾 " + Strings.t("im_farmer", locale), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold) }
        Button(
            onClick = { onChoose("expert") },
            modifier = Modifier.fillMaxWidth().height(64.dp).padding(top = 12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF0EA5E9)),
        ) { Text("🔬 " + Strings.t("im_expert", locale), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold) }

        Row(Modifier.padding(top = 24.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Strings.LOCALES.forEach { (code, label) ->
                LangChip(label, locale == code) { onLocale(code) }
            }
        }
    }
}
