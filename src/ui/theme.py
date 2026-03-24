"""
Custom Gradio theme for RAG Evaluation System.
Modern, professional design with custom color palette.
"""

import gradio as gr


def create_modern_theme() -> gr.themes.Base:
    """
    Create a custom modern theme with professional color scheme.

    Returns:
        Customized Gradio Soft theme
    """
    # Base theme: Soft (clean and modern)
    # Gradio 6.x uses different theme parameters
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    )

    # Apply only the properties supported by Gradio 6.x
    try:
        theme.set(
            # Button styling
            button_primary_background_fill="#3B82F6",
            button_primary_background_fill_hover="#2563EB",
            button_primary_text_color="white",
            # Body background
            body_background_fill="#F9FAFB",
            block_background_fill="white",
            # Typography
            body_text_color="#1F2937",
            # Spacing
            block_radius="8px",
            button_radius="6px",
        )
    except TypeError:
        # If theme.set doesn't work, just return the base theme
        pass

    return theme


# Custom CSS for enhanced styling
CUSTOM_CSS = """
/* ===== Global Container ===== */
.gradio-container {
    max-width: 1400px !important;
    padding: 20px !important;
}

/* ===== Header Styling ===== */
.app-header {
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
}

.app-header h1 {
    color: white !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    margin: 0 0 8px 0 !important;
    display: flex;
    align-items: center;
    gap: 12px;
}

.app-header .subtitle {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 14px !important;
    margin: 0 !important;
}

/* ===== Tab Styling ===== */
.tabs .tab-nav {
    background: #F3F4F6;
    border-radius: 8px;
    padding: 4px;
    margin-bottom: 20px;
}

.tabs .tab-nav button {
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease;
}

.tabs .tab-nav button.selected {
    background: white !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    color: #3B82F6 !important;
}

/* ===== Card/Block Styling ===== */
.gr-box {
    border-radius: 8px !important;
    border: 1px solid #E5E7EB !important;
    background: white !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    transition: box-shadow 0.2s ease;
}

.gr-box:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

/* Section headers */
.gr-box > .markdown-text h3,
.gr-box > .markdown-text h4 {
    color: #1F2937 !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 2px solid #E5E7EB !important;
}

/* ===== Button Styling ===== */
button.primary {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3) !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4) !important;
}

button.secondary {
    background: white !important;
    border: 1px solid #D1D5DB !important;
    color: #374151 !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}

button.secondary:hover {
    background: #F9FAFB !important;
    border-color: #9CA3AF !important;
}

button.stop {
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3) !important;
}

button.stop:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(239, 68, 68, 0.4) !important;
}

/* ===== Input Styling ===== */
input[type="text"],
input[type="password"],
input[type="number"],
textarea,
select {
    border-radius: 6px !important;
    border: 1px solid #D1D5DB !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}

input:focus,
textarea:focus,
select:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    outline: none !important;
}

/* ===== Table/Dataframe Styling ===== */
.dataframe {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #E5E7EB !important;
}

.dataframe table {
    font-size: 14px !important;
}

.dataframe thead {
    background: #F9FAFB !important;
}

.dataframe thead th {
    font-weight: 600 !important;
    color: #374151 !important;
    padding: 12px 16px !important;
    border-bottom: 2px solid #E5E7EB !important;
}

.dataframe tbody tr {
    transition: background 0.15s ease;
}

.dataframe tbody tr:hover {
    background: #F3F4F6 !important;
}

.dataframe tbody tr:nth-child(even) {
    background: #FAFAFA !important;
}

.dataframe tbody tr:nth-child(odd) {
    background: white !important;
}

.dataframe tbody td {
    padding: 12px 16px !important;
    border-top: 1px solid #E5E7EB !important;
}

/* ===== Stat Cards ===== */
.stat-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}

.stat-card:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.stat-card .stat-value {
    font-size: 32px;
    font-weight: 700;
    color: #3B82F6;
    margin-bottom: 4px;
}

.stat-card .stat-label {
    font-size: 14px;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ===== Status Badges ===== */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.status-badge.success {
    background: #D1FAE5;
    color: #059669;
}

.status-badge.warning {
    background: #FEF3C7;
    color: #D97706;
}

.status-badge.error {
    background: #FEE2E2;
    color: #DC2626;
}

.status-badge.info {
    background: #DBEAFE;
    color: #2563EB;
}

/* ===== Score Display ===== */
.score-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
}

.score-excellent {
    background: #D1FAE5;
    color: #059669;
}

.score-good {
    background: #DBEAFE;
    color: #2563EB;
}

.score-fair {
    background: #FEF3C7;
    color: #D97706;
}

.score-poor {
    background: #FEE2E2;
    color: #DC2626;
}

/* ===== Progress Bar ===== */
.progress-bar {
    height: 8px;
    background: #E5E7EB;
    border-radius: 4px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #3B82F6 0%, #1D4ED8 100%);
    border-radius: 4px;
    transition: width 0.3s ease;
}

/* ===== Footer ===== */
.app-footer {
    margin-top: 40px;
    padding-top: 24px;
    border-top: 1px solid #E5E7EB;
    text-align: center;
}

.app-footer p {
    color: #9CA3AF !important;
    font-size: 13px !important;
    margin: 0 !important;
}

.app-footer a {
    color: #3B82F6 !important;
    text-decoration: none !important;
    transition: color 0.2s ease !important;
}

.app-footer a:hover {
    color: #2563EB !important;
    text-decoration: underline !important;
}

/* ===== Markdown Text ===== */
.markdown-text {
    font-size: 14px;
    line-height: 1.6;
}

.markdown-text h1,
.markdown-text h2,
.markdown-text h3,
.markdown-text h4 {
    margin-top: 0 !important;
}

/* ===== Loading Animation ===== */
.loading-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

/* ===== Responsive Design ===== */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px !important;
    }

    .app-header {
        padding: 16px 20px !important;
    }

    .app-header h1 {
        font-size: 22px !important;
    }

    .stat-card .stat-value {
        font-size: 24px;
    }
}
"""