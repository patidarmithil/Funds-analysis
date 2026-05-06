from typing import Optional
from pydantic import BaseModel

class SchemeItem(BaseModel):
    schemeCode: int
    schemeName: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schemeCode": 100027,
                    "schemeName": "Grindlays Super Saver Income Fund-GSSIF-Half Yearly Dividend"
                }
            ]
        }
    }

class NavPoint(BaseModel):
    ds: str    # "YYYY-MM-DD"
    y:  float  # NAV value

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ds": "2023-10-01",
                    "y": 150.25
                }
            ]
        }
    }

class FundResult(BaseModel):
    fund:    str
    code:    int
    data:    list[NavPoint]
    records: int
    error:   Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fund": "Grindlays Super Saver Income Fund-GSSIF-Half Yearly Dividend",
                    "code": 100027,
                    "data": [{"ds": "2023-10-01", "y": 150.25}],
                    "records": 1,
                    "error": None
                }
            ]
        }
    }

class BatchRequest(BaseModel):
    funds:     list[SchemeItem]
    startDate: Optional[str] = None   # "YYYY-MM-DD"
    endDate:   Optional[str] = None   # "YYYY-MM-DD"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "funds": [{"schemeCode": 100027, "schemeName": "Grindlays Super Saver Income Fund-GSSIF-Half Yearly Dividend"}],
                    "startDate": "2023-01-01",
                    "endDate": "2023-10-01"
                }
            ]
        }
    }
