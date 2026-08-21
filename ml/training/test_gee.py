import ee
import traceback
try:
    sa = "inlandroute@inalndroute.iam.gserviceaccount.com"
    cp = r"..\..\backend\secrets\gee_service_account_key.json"
    project = "inalndroute"
    credentials = ee.ServiceAccountCredentials(sa, cp)
    ee.Initialize(credentials=credentials, project=project)
    print("Success")
except Exception as e:
    traceback.print_exc()
