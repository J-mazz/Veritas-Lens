package com.veritaslens

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.veritaslens.model.Article
import com.veritaslens.ui.NewsFeedScreen
import com.veritaslens.ui.theme.VeritasLensTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            VeritasLensTheme {
                val sampleData = listOf(
                    Article("Senate passes major climate bill", "Reuters", 0.32f),
                    Article("White House defends budget proposal", "Fox News", 0.85f),
                    Article("New labor movement rises in tech", "The Guardian", 0.18f)
                )
                NewsFeedScreen(articles = sampleData)
            }
        }
    }
}
