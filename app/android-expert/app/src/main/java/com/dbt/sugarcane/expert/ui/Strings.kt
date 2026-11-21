package com.dbt.sugarcane.expert.ui

/**
 * Lightweight i18n for the Farmer View (and shared chrome).
 *
 * English is fully populated. Hindi ("hi") is intentionally left empty for now and
 * falls back to English, so real Hindi translations can be dropped in later by
 * filling the `hi` map — no structural change required.
 */
object Strings {
    val LOCALES = listOf("en" to "English", "hi" to "हिंदी")

    private val en = mapOf(
        "app_title" to "🌿 Sugarcane Disease",
        "expert_view" to "Expert",
        "farmer_view" to "Farmer",
        "choose_view" to "Welcome — choose your view",
        "choose_view_desc" to "You can switch anytime from the top bar.",
        "im_expert" to "I'm an expert",
        "im_farmer" to "I'm a farmer",
        "language" to "Language",
        "farmer_tagline" to "Point, capture, and check your crop",
        "capture" to "📷 Capture photo",
        "gallery" to "Choose from gallery",
        "analyzing" to "Checking…",
        "analyze" to "Check crop",
        "retake" to "Check another photo",
        "status_healthy" to "Healthy",
        "status_disease" to "%s detected",
        "status_uncertain" to "Uncertain — please recheck",
        "status_possible_redrot" to "Possible Red Rot — please recheck",
        "confidence" to "Confidence",
        "likely" to "Likely",
        "uncertain_word" to "Uncertain",
        "recommended_action" to "Recommended action",
        "consult_expert" to "Please consult an agricultural expert to confirm.",
        "guidance_title" to "📸 Take a clearer photo",
        "guidance_body" to "Take one full-plant photo for screening and a close-up of the affected stalk or leaf, then check again.",
        "redrot_symptoms" to "Red Rot signs",
        "affected_area" to "Affected area",
        "not_sugarcane" to "We couldn't confirm this is a sugarcane image.",
        "analyze_anyway" to "Analyze anyway",
        "server_unreachable" to "Cannot reach the server. Please check your connection.",
    )

    // Pending expert review — empty entries fall back to English.
    private val hi = mapOf<String, String>()

    private val tables = mapOf("en" to en, "hi" to hi)

    fun t(key: String, locale: String, vararg args: Any): String {
        val table = tables[locale] ?: en
        val raw = table[key] ?: en[key] ?: key
        return if (args.isEmpty()) raw else raw.format(*args)
    }
}
