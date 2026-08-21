from functools import cache

from knowledge.service.import_file_service import ImportFileService
from knowledge.service.query_service import QueryService

"""
因为这个类没有状态数据，所以是纯粹的工具服务类（Service/Utility），多个并发请求共享同一个 ImportFileService 实例在线程/协程上是绝对安全的。
使用 @cache 注解将其设为单例模式：
"""
@cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService()

@cache
def get_query_service() -> QueryService:
    return QueryService()