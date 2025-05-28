package com.veritaslens.ui

import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.runtime.Composable
import com.veritaslens.model.Article
import com.veritaslens.ui.components.ArticleCard

@Composable
fun NewsFeedScreen(articles: List<Article>) {
    LazyColumn {
        items(articles) { article ->
            ArticleCard(article)
        }
    }
}
