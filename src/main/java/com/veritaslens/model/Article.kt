package com.veritaslens.model

data class Article(
    val title: String,
    val source: String, // Organization only
    val biasScore: Float // 0.001 to 1.0
)
