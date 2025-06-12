// Example: logic for generating recommendations from analysis results

export function getRecommendations(results) {
    const recommendations = [];
    if (results.fraud_percentage > 10) {
        recommendations.push({
            priority: 'CRITICAL',
            category: 'immediate_action',
            action: 'Implement emergency fraud controls',
            description: 'Fraud rate exceeds 10% - immediate security lockdown recommended'
        });
    } else if (results.fraud_percentage > 5) {
        recommendations.push({
            priority: 'HIGH',
            category: 'security_enhancement',
            action: 'Strengthen fraud detection systems',
            description: 'Deploy additional monitoring and verification steps'
        });
    }
    if (results.high_risk_transactions && results.high_risk_transactions.length > 10) {
        recommendations.push({
            priority: 'HIGH',
            category: 'investigation',
            action: 'Manual review of flagged transactions',
            description: 'Investigate all high-risk transactions for patterns and correlations'
        });
    }
    if (results.total_transactions > 50000) {
        recommendations.push({
            priority: 'MEDIUM',
            category: 'scalability',
            action: 'Consider backend scaling',
            description: 'High transaction volume detected, ensure infrastructure can handle load'
        });
    }
    return recommendations;
}