"""
Main (page-serving) routes.

These endpoints render HTML templates for the browser-based UI.
They are grouped under the ``main`` blueprint with no URL prefix.
"""

from __future__ import annotations

from flask import Blueprint, render_template

main_bp = Blueprint(
    "main",
    __name__,
    template_folder="../templates",
    static_folder="../static",
)


@main_bp.route("/")
def home() -> str:
    """Render the landing / upload page."""
    return render_template("index.html")


@main_bp.route("/predict")
def predict() -> str:
    """Render the Prediction / Classification page."""
    return render_template("predict.html")


@main_bp.route("/about")
def about() -> str:
    """Render the About page."""
    return render_template("about.html")


@main_bp.route("/analytics")
def analytics() -> str:
    """Render the Analytics dashboard page."""
    return render_template("analytics.html")


@main_bp.route("/api-docs")
def api_docs() -> str:
    """Render the interactive API documentation page."""
    return render_template("api_docs.html")


@main_bp.route("/contact")
def contact() -> str:
    """Render the Contact / feedback page."""
    return render_template("contact.html")
