package com.dbt.sugarcane.expert.ui.theme

import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color

/** Colorful green/teal/amber scheme — replaces the default Material purple-on-white. */
val ExpertColors = lightColorScheme(
    primary = Color(0xFF059669),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD1FAE5),
    onPrimaryContainer = Color(0xFF064E3B),
    secondary = Color(0xFF0EA5E9),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFE0F2FE),
    onSecondaryContainer = Color(0xFF0C4A6E),
    tertiary = Color(0xFFF59E0B),
    onTertiary = Color.White,
    background = Color(0xFFF1F7F4),
    onBackground = Color(0xFF0F241F),
    surface = Color.White,
    onSurface = Color(0xFF0F241F),
    surfaceVariant = Color(0xFFE6F2ED),
    onSurfaceVariant = Color(0xFF4B635A),
    error = Color(0xFFEF4444),
)

val HeaderGradient = listOf(Color(0xFF0F766E), Color(0xFF065F46))
val ChipGradient = listOf(Color(0xFF10B981), Color(0xFF0EA5E9))
