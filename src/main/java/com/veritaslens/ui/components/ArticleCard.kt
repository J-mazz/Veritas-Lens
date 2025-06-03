@Composable
fun ArticleCard(article: Article) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // Article title (main focus)
            Text(
                text = article.title,
                style = MaterialTheme.typography.titleMedium
            )
            // Bias rating bar with clear label
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp)
                    .background(color = biasColor(article.biasScore))
            ) {
                Spacer(modifier = Modifier.weight(1f))
                Text(
                    text = biasLabel(article.biasScore),
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(end = 4.dp)
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            // Attribution (organization)
            Text(
                text = article.source,
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

// Helper functions
private fun biasColor(score: Float): Color {
    return when {
        score < 0.33 -> Color.Blue.copy(alpha = 0.5f)    // Left
        score < 0.67 -> Color.Gray.copy(alpha = 0.5f)    // Objective/Center
        else -> Color.Red.copy(alpha = 0.5f)             // Right
    }
}
private fun biasLabel(score: Float): String {
    return when {
        score < 0.33 -> "Left"
        score < 0.67 -> "Center"
        else -> "Right"
    }
}
