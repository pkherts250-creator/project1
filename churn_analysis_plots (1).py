import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# NOTE: `customer_churn_df` must already exist before running the plotting code.
# Load or build it here, e.g.:
# customer_churn_df = pd.read_csv("customer_churn.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

tier_churn = customer_churn_df.groupby("subscription_tier")["churn"].agg(["count", "mean"]).reset_index()
sns.barplot(data=tier_churn, x="subscription_tier", y="mean", ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Churn Rate by Subscription Tier", fontweight="bold")
axes[0, 0].set_ylabel("Churn Rate")
axes[0, 0].set_xlabel("Subscription Tier")
for idx, row in tier_churn.iterrows():
    axes[0, 0].text(idx, row["mean"] + 0.015, f"{row['mean']*100:.1f}% (n={int(row['count'])})", ha="center", fontsize=10)
axes[0, 0].set_ylim(0, 0.6)

sns.boxplot(data=customer_churn_df, x="churn", y="login_velocity_change", ax=axes[0, 1], palette="Set2")
axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0, 1].set_title("30-Day Login Velocity Change vs. Churn", fontweight="bold")
axes[0, 1].set_xticklabels(["Retained (0)", "Churned (1)"])
axes[0, 1].set_ylabel("Login Change (Last 30d - Prior 30d)")

sns.boxplot(data=customer_churn_df, x="churn", y="order_velocity_change", ax=axes[1, 0], palette="Set2")
axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[1, 0].set_title("30-Day Order Velocity Change vs. Churn", fontweight="bold")
axes[1, 0].set_xticklabels(["Retained (0)", "Churned (1)"])
axes[1, 0].set_ylabel("Order Change (Last 30d - Prior 30d)")

exclude_meta = {
    "customer_id", "active_tenure_days", "active_tenure_months", "total_orders",
    "physical_orders", "subscription_renewals", "total_spend_gbp", "total_physical_units",
    "review_count", "low_rating_count", "late_deliveries", "crm_total_contacts",
    "crm_complaints", "crm_queries", "crm_email_count", "crm_phone_count",
    "crm_chat_count", "total_logins", "unique_ips"
}
num_features = [c for c in customer_churn_df.select_dtypes(include=[np.number]).columns if c not in exclude_meta]
corrs = customer_churn_df[num_features].corr()["churn"].drop("churn").sort_values()
top_corrs = pd.concat([corrs.head(6), corrs.tail(6)])
top_corrs.plot(kind="barh", ax=axes[1, 1], color=["#d95f02" if v < 0 else "#1b9e77" for v in top_corrs])
axes[1, 1].set_title("Top Positive & Negative Correlations with Churn", fontweight="bold")
axes[1, 1].set_xlabel("Pearson Correlation Coefficient (r)")

plt.tight_layout()
plt.show()
