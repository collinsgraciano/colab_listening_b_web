"""Shared Jinja2Templates instance + custom filters."""
from datetime import datetime

from fastapi.templating import Jinja2Templates

from .paths import TEMPLATES_DIR

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Custom Jinja2 filter for date formatting
def _datestr(ts):
    if not ts:
        return ""
    try:
        d = datetime.fromtimestamp(float(ts))
        now = datetime.now()
        diff = (now - d).total_seconds()
        if diff < 3600:
            return f"{int(diff/60)}分钟前"
        elif diff < 86400:
            return f"{int(diff/3600)}小时前"
        elif diff < 604800:
            return f"{int(diff/86400)}天前"
        else:
            return d.strftime("%m-%d")
    except (ValueError, TypeError):
        return ""


templates.env.filters["datestr"] = _datestr
