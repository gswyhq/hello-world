import traceback
from typing import Dict, List, Any, Union
import json
import sqlparse
from sqlparse.sql import Comparison, Parenthesis, IdentifierList, Identifier, Token
from sqlparse.tokens import Whitespace, Keyword, Punctuation, String, Number

import requests
from typing import Dict, Any

from typing import Dict, Any, List

import requests
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

false = False
true = True
null = None

def query_elasticsearch(query: Dict[str, Any], es_url: str = "http://localhost:9200", index_name: str = "test_users") -> Dict[str, Any]:
    """
    使用 requests 向 Elasticsearch 发送查询请求。

    参数:
        query (dict): 符合 Elasticsearch DSL 的查询语句。
        es_url (str): Elasticsearch 服务地址，默认为 http://localhost:9200。
        index_name (str): 要查询的索引名称。

    返回:
        dict: Elasticsearch 返回的原始 JSON 响应。

    异常:
        requests.exceptions.RequestException: 网络或请求错误。
        ValueError: 返回非 2xx 状态码。
    """
    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }

    url = f"{es_url}/{index_name}/_search"

    try:
        response = requests.post(url, json=query, headers=headers)
        # response.raise_for_status()  # 如果状态码不是 2xx，抛出异常
        return response.json()
    except Exception as e:
        print(f"请求 Elasticsearch 失败: {e}")
        raise


# ========================
# 字段类型常量（便于维护）
# ========================
FIELD_TYPE_KEYWORD = "keyword"
FIELD_TYPE_TEXT = "text"
FIELD_TYPE_DATE = "date"
FIELD_TYPE_INTEGER = "integer"
FIELD_TYPE_LONG = "long"
FIELD_TYPE_FLOAT = "float"
FIELD_TYPE_DOUBLE = "double"

# 可聚合的类型（支持 terms 聚合）
AGGREGATABLE_TYPES = {
    FIELD_TYPE_KEYWORD,
    FIELD_TYPE_INTEGER,
    FIELD_TYPE_LONG,
    FIELD_TYPE_FLOAT,
    FIELD_TYPE_DOUBLE,
    FIELD_TYPE_DATE,
}

DEFAULT_FIELD_TYPES = {'active': 'integer',
     'age': 'integer',
     'code': 'keyword',
     'created_at': 'date',
     'created_date': 'date',
     'department': 'keyword',
     'email': 'keyword',
     'login_time': 'date',
     'name': 'keyword',
     'register_time': 'date',
     'salary': 'float',
     'status': 'keyword',
     'updated_at': 'date',
     'x': 'integer',
     'y': 'integer',
     'z': 'integer',
     '会员等级': 'keyword',
     '创建时间': 'date',
     '学历': 'keyword',
     '性别': 'keyword',
     '最后登录时间': 'date',
     '标签': 'keyword',
     '活跃状态': 'keyword',
     '消费金额': 'float',
     '登录次数': 'integer',
     '籍贯': 'keyword',
     '职业': 'keyword',
     '设备类型': 'keyword',
     '访问渠道': 'keyword',
     '邮箱': 'keyword'
                       }


# 需要 .keyword 子字段才能精确匹配/聚合的类型
REQUIRES_KEYWORD_SUBFIELD = {FIELD_TYPE_TEXT}


class ESQueryGenerator:
    """Elasticsearch 查询生成器（兼容 ES 5.4+，支持显式字段类型）"""

    def __init__(self):
        pass

    def _get_field_for_term_query(self, field: str, field_type: str) -> str:
        """
        根据字段类型返回用于 term/terms/wildcard 查询的实际字段名
        """
        if field_type in REQUIRES_KEYWORD_SUBFIELD:
            return f"{field}.keyword"
        else:
            return field  # keyword / date / numeric 直接使用原字段

    def _get_field_for_agg(self, field: str, field_type: str) -> str:
        """
        根据字段类型返回用于 terms 聚合的实际字段名
        注意：text 字段必须用 .keyword；其他类型直接用原字段
        """
        if field_type in REQUIRES_KEYWORD_SUBFIELD:
            return f"{field}.keyword"
        elif field_type in AGGREGATABLE_TYPES:
            return field
        else:
            raise ValueError(f"字段 '{field}' 类型为 '{field_type}'，不支持 terms 聚合")

    def _build_condition(
            self,
            condition: Dict[str, Any],
            field_types=DEFAULT_FIELD_TYPES,
    ) -> Dict[str, Any]:
        """
        递归构建查询条件，依赖 field_types 判断字段真实类型
        """
        condition_type = condition["type"]
        field = condition.get("field", "")

        # if field and field not in field_types:
        #     raise ValueError(f"字段 '{field}' 未在 field_types 中定义类型")

        field_type = field_types.get(field, FIELD_TYPE_KEYWORD)  # 默认 fallback

        if condition_type == "and":
            return {
                "bool": {
                    "must": [self._build_condition(c, field_types) for c in condition["conditions"]]
                }
            }
        elif condition_type == "or":
            return {
                "bool": {
                    "should": [self._build_condition(c, field_types) for c in condition["conditions"]],
                    "minimum_should_match": 1
                }
            }
        elif condition_type == "not":
            return {
                "bool": {
                    "must_not": [self._build_condition(condition["condition"], field_types)]
                }
            }
        elif condition_type == "equal":
            actual_field = self._get_field_for_term_query(field, field_type)
            return {"term": {actual_field: condition["value"]}}
        elif condition_type == "not_equal":
            actual_field = self._get_field_for_term_query(field, field_type)
            return {"bool": {"must_not": {"term": {actual_field: condition["value"]}}}}
        elif condition_type in ("greater", "less", "greater_equal", "less_equal"):
            # range 查询适用于 date / numeric，不适用于 text/keyword（除非是 numeric keyword）
            op_map = {
                "greater": "gt",
                "less": "lt",
                "greater_equal": "gte",
                "less_equal": "lte"
            }
            return {"range": {field: {op_map[condition_type]: condition["value"]}}}
        elif condition_type == "in":
            actual_field = self._get_field_for_term_query(field, field_type)
            return {"terms": {actual_field: condition["values"]}}
        elif condition_type == "not_in":
            actual_field = self._get_field_for_term_query(field, field_type)
            return {"bool": {"must_not": {"terms": {actual_field: condition["values"]}}}}
        elif condition_type == "contains":
            # 注意：wildcard 只适用于 keyword 或 text.keyword，不能用于 date/numeric
            if field_type not in (FIELD_TYPE_KEYWORD, FIELD_TYPE_TEXT):
                raise ValueError(f"字段 '{field}' 类型为 '{field_type}'，不支持 contains 操作")
            actual_field = self._get_field_for_term_query(field, field_type)
            return {"wildcard": {actual_field: f"*{condition['value']}*"}}
        elif condition_type == "not_contains":
            if field_type not in (FIELD_TYPE_KEYWORD, FIELD_TYPE_TEXT):
                raise ValueError(f"字段 '{field}' 类型为 '{field_type}'，不支持 not_contains 操作")
            actual_field = self._get_field_for_term_query(field, field_type)
            return {"bool": {"must_not": {"wildcard": {actual_field: f"*{condition['value']}*"}}}}
        elif condition_type.lower() == "like":
            if field_type not in (FIELD_TYPE_KEYWORD, FIELD_TYPE_TEXT):
                raise ValueError(f"字段 '{field}' 类型为 '{field_type}'，不支持 LIKE 操作")
            pattern = condition["pattern"].replace('%', '*').replace('_', '?')
            actual_field = self._get_field_for_term_query(field, field_type)
            return {"wildcard": {actual_field: pattern}}
        else:
            raise ValueError(f"不支持的条件类型: {condition_type}")

    def generate_query(
            self,
            filter_conditions: Dict[str, Any],
            analysis_fields: List[str],
            field_types: Dict[str, str] = DEFAULT_FIELD_TYPES,
            aggs_size: int = 100
    ) -> Dict[str, Any]:
        """
        生成完整的 Elasticsearch 查询语句

        Args:
            filter_conditions: 过滤条件树
            analysis_fields: 需要聚合分析的字段列表
            field_types: 字段名 -> 类型 的映射，例如 {"年龄": "integer", "城市": "keyword"}
            aggs_size: 聚合返回的桶数量
        """
        query = {
            "size": 0,
            # "track_total_hits": False,
            "query": {"bool": {}},
            "aggs": {}
        }

        # 构建查询条件
        built_cond = self._build_condition(filter_conditions, field_types)
        query["query"]["bool"] = built_cond.get("bool", {})

        # 构建聚合
        for field in analysis_fields:
            # if field not in field_types:
            #     raise ValueError(f"聚合字段 '{field}' 未在 field_types 中定义类型")
            field_type = field_types.get(field, FIELD_TYPE_KEYWORD)
            agg_field = self._get_field_for_agg(field, field_type)
            query["aggs"][f"{field}_分布"] = {
                "terms": {
                    "field": agg_field,
                    "size": aggs_size
                }
            }

        return query

generator = ESQueryGenerator()


# ========================
# SQL WHERE 子句解析器（增强版）
# ========================

def _clean_value(val: str):
    val = val.strip()
    if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
        return val[1:-1]
    try:
        if '.' in val:
            return float(val)
        else:
            return int(val)
    except ValueError:
        return val


def _is_keyword(token, keywords):
    if not isinstance(token, Token):
        return False
    return token.ttype is Keyword and token.value.upper() in (k.upper() for k in keywords)


def _extract_in_values(paren_token: Parenthesis):
    values = []
    inner_tokens = [t for t in paren_token.tokens if not t.is_whitespace and str(t).strip() not in '()']
    if not inner_tokens:
        return values
    for t in inner_tokens:
        if isinstance(t, IdentifierList):
            for sub in t.get_identifiers():
                values.append(_clean_value(str(sub)))
        elif isinstance(t, Identifier) or t.ttype in [String.Single, Number.Integer, Number.Float]:
            values.append(_clean_value(str(t)))
        elif str(t).strip() not in [',']:
            values.append(_clean_value(str(t)))
    return values


def _parse_comparison(comp: Comparison):
    tokens = [t for t in comp.tokens if not t.is_whitespace]
    if len(tokens) < 3:
        raise ValueError(f"无效 Comparison: {comp}")

    # === 修复核心：直接通过字符串值识别 LIKE，不依赖 token 类型 ===
    if len(tokens) == 3:
        token_str = str(tokens[1]).strip().upper()
        if token_str == "LIKE":
            field = "".join(str(t).strip() for t in tokens[0])
            pattern = _clean_value(str(tokens[2]))
            return {
                "type": "like",
                "field": field,
                "pattern": pattern
            }

    left = str(tokens[0]).strip()

    # IN
    if len(tokens) >= 3 and _is_keyword(tokens[1], ["IN"]) and isinstance(tokens[2], Parenthesis):
        return {
            "type": "in",
            "field": left,
            "values": _extract_in_values(tokens[2])
        }

    # NOT IN
    if (len(tokens) >= 4 and
        _is_keyword(tokens[1], ["NOT"]) and
        _is_keyword(tokens[2], ["IN"]) and
        isinstance(tokens[3], Parenthesis)):
        return {
            "type": "not",
            "condition": {
                "type": "in",
                "field": left,
                "values": _extract_in_values(tokens[3])
            }
        }

    # LIKE（备用路径，以防上面未命中）
    if len(tokens) == 3 and _is_keyword(tokens[1], ["LIKE"]):
        return {
            "type": "like",
            "field": left,
            "pattern": _clean_value(str(tokens[2]))
        }

    # 二元操作符
    if len(tokens) == 3:
        op_raw = str(tokens[1]).strip().upper()
        op_map = {
            "=": "equal",
            "!=": "not_equal",
            "<>": "not_equal",
            ">": "greater",
            ">=": "greater_equal",
            "<": "less",
            "<=": "less_equal"
        }
        if op_raw in op_map:
            return {
                "type": op_map[op_raw],
                "field": left,
                "value": _clean_value(str(tokens[2]))
            }

    raise ValueError(f"无法解析 Comparison: {comp}")


def _parse_atomic_condition_from_tokens(tokens):
    tokens = [t for t in tokens if not t.is_whitespace]
    if len(tokens) == 3:
        a, op, b = tokens
        if isinstance(a, (Identifier, Token)) and _is_keyword(op, ["IN"]) and isinstance(b, Parenthesis):
            return {
                "type": "in",
                "field": str(a).strip(),
                "values": _extract_in_values(b)
            }
        # 新增 LIKE 支持
        if isinstance(a, (Identifier, Token)) and _is_keyword(op, ["LIKE"]):
            return {
                "type": "like",
                "field": str(a).strip(),
                "pattern": _clean_value(str(b))
            }
    if len(tokens) == 4:
        a, not_kw, in_kw, b = tokens
        if (isinstance(a, (Identifier, Token)) and
            _is_keyword(not_kw, ["NOT"]) and
            _is_keyword(in_kw, ["IN"]) and
            isinstance(b, Parenthesis)):
            return {
                "type": "not",
                "condition": {
                    "type": "in",
                    "field": str(a).strip(),
                    "values": _extract_in_values(b)
                }
            }
    if len(tokens) == 3:
        a, op, b = tokens
        if isinstance(a, (Identifier, Token)) and isinstance(b, (Token, Identifier)):
            op_str = str(op).strip().upper()
            op_map = {
                "=": "equal",
                "!=": "not_equal",
                "<>": "not_equal",
                ">": "greater",
                ">=": "greater_equal",
                "<": "less",
                "<=": "less_equal"
            }
            if op_str in op_map:
                return {
                    "type": op_map[op_str],
                    "field": str(a).strip(),
                    "value": _clean_value(str(b))
                }
    return None


def _parse_expression(tokens):
    tokens = [t for t in tokens if not t.is_whitespace]
    if not tokens:
        return None

    if len(tokens) == 1 and isinstance(tokens[0], Comparison):
        return _parse_comparison(tokens[0])

    if len(tokens) == 1 and isinstance(tokens[0], Parenthesis):
        return _parse_expression(tokens[0].tokens[1:-1])

    if _is_keyword(tokens[0], ["NOT"]):
        rest = tokens[1:]
        if len(rest) == 1 and isinstance(rest[0], Parenthesis):
            inner = _parse_expression(rest[0].tokens[1:-1])
        else:
            inner = _parse_expression(rest)
        return {"type": "not", "condition": inner}

    level = 0
    candidates = []
    for idx, token in enumerate(tokens):
        if token.match(Punctuation, '('):
            level += 1
        elif token.match(Punctuation, ')'):
            level -= 1
        elif level == 0 and _is_keyword(token, ["AND", "OR"]):
            candidates.append((idx, token.value.lower()))

    if candidates:
        op_type = candidates[0][1]
        conditions = []
        start = 0
        for idx, _ in candidates:
            cond = _parse_expression(tokens[start:idx])
            if cond:
                conditions.append(cond)
            start = idx + 1
        cond = _parse_expression(tokens[start:])
        if cond:
            conditions.append(cond)
        if len(conditions) == 1:
            return conditions[0]
        return {"type": op_type, "conditions": conditions}

    atomic = _parse_atomic_condition_from_tokens(tokens)
    if atomic:
        return atomic

    raw = "".join(str(t) for t in tokens)
    try:
        parsed = sqlparse.parse(raw)[0]
        if len(parsed.tokens) == 1 and isinstance(parsed.tokens[0], Comparison):
            return _parse_comparison(parsed.tokens[0])
    except Exception:
        pass

    raise ValueError(f"无法解析表达式: {' '.join(str(t) for t in tokens)}")


def sql_to_filter_conditions(where_clause: str):
    where_clause = where_clause.strip()
    if not where_clause:
        raise ValueError("WHERE 子句不能为空")

    parsed = sqlparse.parse(where_clause)
    if not parsed:
        raise ValueError("SQL 解析失败")

    inner_cond = _parse_expression(parsed[0].tokens)

    if inner_cond is None:
        raise ValueError("解析结果为空")

    if isinstance(inner_cond, dict) and inner_cond.get("type") in ("and", "or"):
        return inner_cond

    return {"type": "and", "conditions": [inner_cond]}

# ========================
# 测试用例
# ========================

def test_cases2():
    """测试 SQL 解析 + ES 查询生成端到端流程"""
    test_cases = [
        " age > 18",
        " (age > 18)",
        " age > 18 AND salary < 5000",
        " name = 'Alice' OR name = 'Bob'",
        " age > 18 AND (department = 'IT' OR department = 'HR')",
        " (age > 18 AND salary > 5000) OR (age <= 18 AND salary > 3000)",
        " salary not in ( 3,4,5,6)",
        " (age > 18 AND name = 'Alice') OR ((salary >= 5000 or salary < -500 ) AND department IN ('IT', 'HR'))",
        " (status = 'A' OR status = 'B') AND (active = 1 OR last_login > '2023-01-01')",
        " (x = 1) OR y = 2 OR z != 3",
        " (x = 1) OR ((y = 2) OR (z != 3))",
        " NOT active = 0 AND age > 20",
        "(age > 18) AND (name = 'Alice' OR salary in ( 5000 , 10000))",
        "age > 18 AND (name = 'Alice' OR (salary >= 5000 AND salary < 10000))",
        " age > 18 ",
        " name LIKE 'A%'",
        " email LIKE '%@qq.com'",
        " code LIKE 'A_B'",  # _ 匹配单字符
        " created_at > '2023-01-01'",
        " login_time <= '2024-12-31T23:59:59'",
        " status = 'active' AND created_date >= '2023-01-01'",
        " (name LIKE 'John%') OR (email LIKE '%gmail.com')",
        " NOT updated_at < '2020-01-01'",
        " register_time IN ('2023-01-01', '2023-01-02')",
        " last_login LIKE '2023%' AND age > 25"
    ]

    for i, sql in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        print("SQL:", sql)
        try:
            cond = sql_to_filter_conditions(sql)
            print(json.dumps(cond, indent=2, ensure_ascii=False))
            analysis_fields1 = ["age", "婚姻状态", "学历"]
            query1 = generator.generate_query(cond, analysis_fields1)
            print("生成的 ES 查询:")
            print(json.dumps(query1, indent=2, ensure_ascii=False))

            rets = query_elasticsearch(query1)
            print(f"es查询结果：{json.dumps(rets, indent=2, ensure_ascii=False)}")
            print("\n" + "=" * 50 + "\n")
        except Exception as e:
            print("❌ 错误:", str(e))
            print(f"错误详情：{traceback.format_exc()}")


def test_cases():
    """手动构造条件树的测试用例"""
    # 测试用例1: 简单条件组合
    filter_cond_list = [{
        "type": "and",
        "conditions": [
            {"type": "equal", "field": "籍贯", "value": "广东"},
            {"type": "equal", "field": "性别", "value": "女"},
            {"type": "greater", "field": "age", "value": 18},
            {"type": "less", "field": "age", "value": 60}
        ]
    }, {
        "type": "and",
        "conditions": [
            {
                "type": "or",
                "conditions": [
                    {"type": "equal", "field": "籍贯", "value": "广东"},
                    {"type": "equal", "field": "籍贯", "value": "广西"}
                ]
            },
            {
                "type": "not",
                "condition": {
                    "type": "in",
                    "field": "学历",
                    "values": ["博士", "硕士"]
                }
            },
            {
                "type": "and",
                "conditions": [
                    {"type": "greater", "field": "消费金额", "value": 1000},
                    {"type": "contains", "field": "邮箱", "value": "@qq.com"}
                ]
            }
        ]
    },{
        "type": "and",
        "conditions": [
            {"type": "equal", "field": "活跃状态", "value": "是"},
            {
                "type": "not",
                "condition": {
                    "type": "or",
                    "conditions": [
                        {"type": "less", "field": "登录次数", "value": 5},
                        {"type": "contains", "field": "标签", "value": "黑名单"}
                    ]
                }
            }
        ]
    }, {
        "type": "and",
        "conditions": [
            # 籍贯 in (广东, 广西)
            {
                "type": "in",
                "field": "籍贯",
                "values": ["广东", "广西"]
            },
            # (性别=女) or (职业!=教师)
            {
                "type": "or",
                "conditions": [
                    {"type": "equal", "field": "性别", "value": "女"},
                    {"type": "not_equal", "field": "职业", "value": "教师"}
                ]
            }
        ]
    },
        {
            "type": "and",
            "conditions": [
                {"type": "like", "field": "邮箱", "pattern": "%@example.com"},
                {"type": "greater_equal", "field": "创建时间", "value": "2023-01-01"},
                {"type": "less", "field": "最后登录时间", "value": "2025-01-01"}
            ]
        }
    ]
    analysis_fields4 = ["设备类型", "访问渠道", "会员等级"]
    for cond in filter_cond_list:
        print('*'*80)
        print(f"条件：{cond}")
        query4 = generator.generate_query(cond, analysis_fields4)
        print(json.dumps(query4, indent=2, ensure_ascii=False))


# if __name__ == "__main__":
#     test_cases()
#     test_cases2()


# ========================
# FastAPI 模型与路由
# ========================
app = FastAPI(title="SQL to Elasticsearch Query API", version="1.0")

class QueryRequest(BaseModel):
    sql_where: str
    analysis_fields: List[str]
    index_name: Optional[str] = "test_users"
    es_url: Optional[str] = "http://localhost:9200"

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        # 解析 SQL WHERE
        filter_conditions = sql_to_filter_conditions(request.sql_where)

        # 生成 ES 查询
        es_query = generator.generate_query(
            filter_conditions=filter_conditions,
            analysis_fields=request.analysis_fields,
            field_types=DEFAULT_FIELD_TYPES
        )

        # 查询 ES
        es_result = query_elasticsearch(es_query, es_url=request.es_url, index_name=request.index_name)

        return {
            "success": True,
            "es_query": es_query,
            "es_result": es_result
        }

    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc()
        })

# ========================
# 直接运行入口（关键！）
# ========================
if __name__ == "__main__":
    import uvicorn
    print("🚀 正在启动 FastAPI 应用...")
    print("访问文档: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)

# 测试请求示例：
'''
curl -XPOST http://localhost:8000/query -H "Content-Type:application/json; charset=utf-8" -d '
{
  "sql_where": "age > 18 AND name LIKE 'A'",
  "analysis_fields": ["职业", "status", "职业", "籍贯"]
}'



docker run -d \
  --name es543 \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "bootstrap.memory_lock=true" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  docker.elastic.co/elasticsearch/elasticsearch:5.4.3


curl -XPUT "http://localhost:9200/test_users" -H "Content-Type: application/json" -d'
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "user": {
      "properties": {
        "age": { "type": "integer" },
        "salary": { "type": "float" },
        "name": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
        "email": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
        "code": { "type": "keyword" },
        "department": { "type": "keyword" },
        "status": { "type": "keyword" },
        "active": { "type": "integer" },
        "x": { "type": "integer" },
        "y": { "type": "integer" },
        "z": { "type": "integer" },
        "籍贯": { "type": "keyword" },
        "性别": { "type": "keyword" },
        "学历": { "type": "keyword" },
        "消费金额": { "type": "float" },
        "邮箱": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
        "活跃状态": { "type": "keyword" },
        "登录次数": { "type": "integer" },
        "标签": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
        "职业": { "type": "keyword" },
        "设备类型": { "type": "keyword" },
        "访问渠道": { "type": "keyword" },
        "会员等级": { "type": "keyword" },
        "created_at": { "type": "date", "format": "yyyy-MM-dd||yyyy-MM-dd HH:mm:ss||epoch_millis" },
        "login_time": { "type": "date", "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis" },
        "created_date": { "type": "date", "format": "yyyy-MM-dd" },
        "updated_at": { "type": "date", "format": "yyyy-MM-dd" },
        "register_time": { "type": "date", "format": "yyyy-MM-dd" },
        "最后登录时间": { "type": "date", "format": "yyyy-MM-dd" },
        "创建时间": { "type": "date", "format": "yyyy-MM-dd" }
      }
    }
  }
}'

curl -XPOST "http://localhost:9200/test_users/user/_bulk?pretty" -H "Content-Type: application/json" --data-binary '
{"index":{}}
{"age":25,"salary":4500.0,"name":"Alice","email":"alice@qq.com","code":"A1B","department":"IT","status":"active","active":1,"x":1,"y":2,"z":4,"籍贯":"广东","性别":"女","学历":"本科","消费金额":1200.5,"邮箱":"alice@example.com","活跃状态":"是","登录次数":10,"标签":"VIP","职业":"工程师","设备类型":"手机","访问渠道":"APP","会员等级":"黄金","created_at":"2023-05-10","login_time":"2024-10-01 10:30:00","created_date":"2023-01-15","updated_at":"2023-06-01","register_time":"2023-01-01","最后登录时间":"2024-10-01","创建时间":"2023-01-10"}
{"index":{}}
{"age":30,"salary":8000.0,"name":"Bob","email":"bob@gmail.com","code":"A2B","department":"HR","status":"A","active":1,"x":0,"y":2,"z":3,"籍贯":"广西","性别":"男","学历":"硕士","消费金额":3000.0,"邮箱":"bob@gmail.com","活跃状态":"是","登录次数":20,"标签":"普通","职业":"教师","设备类型":"PC","访问渠道":"Web","会员等级":"白银","created_at":"2022-12-01","login_time":"2024-11-01 09:00:00","created_date":"2022-12-01","updated_at":"2024-01-01","register_time":"2023-01-02","最后登录时间":"2024-11-01","创建时间":"2022-12-01"}
{"index":{}}
{"age":17,"salary":3500.0,"name":"Charlie","email":"charlie@qq.com","code":"C3D","department":"IT","status":"B","active":0,"x":1,"y":1,"z":3,"籍贯":"湖南","性别":"男","学历":"博士","消费金额":800.0,"邮箱":"charlie@qq.com","活跃状态":"否","登录次数":3,"标签":"黑名单","职业":"学生","设备类型":"平板","访问渠道":"H5","会员等级":"普通","created_at":"2024-01-01","login_time":"2024-02-01","created_date":"2024-01-01","updated_at":"2020-01-01","register_time":"2024-01-01","最后登录时间":"2024-02-01","创建时间":"2024-01-01"}
{"index":{}}
{"age":40,"salary":-1000.0,"name":"David","email":"david@example.com","code":"A_B","department":"Finance","status":"inactive","active":1,"x":1,"y":2,"z":5,"籍贯":"广东","性别":"男","学历":"大专","消费金额":5000.0,"邮箱":"david@example.com","活跃状态":"是","登录次数":15,"标签":"VIP","职业":"会计","设备类型":"手机","访问渠道":"APP","会员等级":"钻石","created_at":"2023-03-01","login_time":"2024-12-31 23:59:59","created_date":"2023-03-01","updated_at":"2023-04-01","register_time":"2023-03-01","最后登录时间":"2024-12-31","创建时间":"2023-03-01"}
{"index":{}}
{"age":22,"salary":6000.0,"name":"Eva","email":"eva@163.com","code":"E5F","department":"IT","status":"active","active":1,"x":0,"y":0,"z":0,"籍贯":"广西","性别":"女","学历":"本科","消费金额":2000.0,"邮箱":"eva@163.com","活跃状态":"是","登录次数":8,"标签":"普通","职业":"设计师","设备类型":"PC","访问渠道":"Web","会员等级":"黄金","created_at":"2023-07-01","login_time":"2024-09-01","created_date":"2023-07-01","updated_at":"2023-08-01","register_time":"2023-07-01","最后登录时间":"2024-09-01","创建时间":"2023-07-01"}
'

curl -XPOST "http://localhost:9200/test_users/user/_bulk?pretty" -H "Content-Type: application/json" --data-binary '
{"index":{}}
{"age":19,"salary":5000.0,"name":"Frank","email":"frank@example.com","code":"F7G","department":"IT","status":"active","active":1,"x":1,"y":2,"z":3,"籍贯":"广东","性别":"男","学历":"大专","消费金额":2500.0,"邮箱":"frank@example.com","活跃状态":"是","登录次数":12,"标签":"新用户,学生","职业":"实习生","设备类型":"手机","访问渠道":"APP","会员等级":"普通","created_at":"2023-01-01","login_time":"2024-05-10 14:20:00","created_date":"2023-01-01","updated_at":"2023-02-01","register_time":"2023-01-01","最后登录时间":"2024-05-10","创建时间":"2023-01-01"}
{"index":{}}
{"age":60,"salary":10000.0,"name":"Grace","email":"grace@gmail.com","code":"G8H","department":"HR","status":"B","active":1,"x":0,"y":0,"z":4,"籍贯":"广西","性别":"女","学历":"博士","消费金额":8000.0,"邮箱":"grace@gmail.com","活跃状态":"是","登录次数":30,"标签":"高管","职业":"总监","设备类型":"PC","访问渠道":"Web","会员等级":"钻石","created_at":"2022-11-15","login_time":"2024-12-30 18:00:00","created_date":"2022-11-15","updated_at":"2024-01-10","register_time":"2022-11-15","最后登录时间":"2024-12-30","创建时间":"2022-11-15"}
{"index":{}}
{"age":16,"salary":2000.0,"name":"Henry","email":"henry@qq.com","code":"H9I","department":"Support","status":"inactive","active":0,"x":1,"y":1,"z":1,"籍贯":"湖南","性别":"男","学历":"高中","消费金额":300.0,"邮箱":"henry@qq.com","活跃状态":"否","登录次数":2,"标签":"黑名单,未成年","职业":"学生","设备类型":"平板","访问渠道":"H5","会员等级":"普通","created_at":"2024-03-01","login_time":"2024-04-01","created_date":"2024-03-01","updated_at":"2020-01-01","register_time":"2024-03-01","最后登录时间":"2024-04-01","创建时间":"2024-03-01"}
{"index":{}}
{"age":28,"salary":-500.0,"name":"Ivy","email":"ivy@163.com","code":"I_J","department":"IT","status":"A","active":1,"x":1,"y":2,"z":6,"籍贯":"广东","性别":"女","学历":"本科","消费金额":4000.0,"邮箱":"ivy@163.com","活跃状态":"是","登录次数":25,"标签":"VIP","职业":"程序员","设备类型":"Mac","访问渠道":"APP","会员等级":"黄金","created_at":"2023-08-20","login_time":"2024-11-15 09:45:00","created_date":"2023-08-20","updated_at":"2023-09-01","register_time":"2023-08-20","最后登录时间":"2024-11-15","创建时间":"2023-08-20"}
{"index":{}}
{"age":35,"salary":3000.0,"name":"Jack","email":"jack@example.com","code":"J1K","department":"Finance","status":"active","active":1,"x":0,"y":2,"z":3,"籍贯":"四川","性别":"男","学历":"硕士","消费金额":6000.0,"邮箱":"jack@example.com","活跃状态":"是","登录次数":18,"标签":"普通","职业":"教师","设备类型":"PC","访问渠道":"Web","会员等级":"白银","created_at":"2023-02-14","login_time":"2024-10-20","created_date":"2023-02-14","updated_at":"2023-03-01","register_time":"2023-02-14","最后登录时间":"2024-10-20","创建时间":"2023-02-14"}
{"index":{}}
{"age":21,"salary":7000.0,"name":"Kate","email":"kate@gmail.com","code":"K2L","department":"IT","status":"B","active":1,"x":1,"y":0,"z":5,"籍贯":"广西","性别":"女","学历":"本科","消费金额":3500.0,"邮箱":"kate@gmail.com","活跃状态":"是","登录次数":7,"标签":"学生","职业":"研究生","设备类型":"手机","访问渠道":"APP","会员等级":"普通","created_at":"2023-09-01","login_time":"2024-08-01","created_date":"2023-09-01","updated_at":"2023-10-01","register_time":"2023-09-01","最后登录时间":"2024-08-01","创建时间":"2023-09-01"}
{"index":{}}
{"age":45,"salary":12000.0,"name":"Leo","email":"leo@outlook.com","code":"L3M","department":"Executive","status":"active","active":1,"x":1,"y":1,"z":2,"籍贯":"北京","性别":"男","学历":"博士","消费金额":15000.0,"邮箱":"leo@outlook.com","活跃状态":"是","登录次数":40,"标签":"高管,VIP","职业":"CTO","设备类型":"PC","访问渠道":"Web","会员等级":"钻石","created_at":"2022-01-01","login_time":"2024-12-31 23:59:59","created_date":"2022-01-01","updated_at":"2023-12-01","register_time":"2022-01-01","最后登录时间":"2024-12-31","创建时间":"2022-01-01"}
'


curl -XPOST "http://localhost:9200/test_users/user/_search?pretty" -H "Content-Type: application/json" -d'
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        { "range": { "age": { "gt": 18 } } },
        {
          "bool": {
            "should": [
              { "term": { "name.keyword": "Alice" } },
              { "terms": { "salary": [5000, 10000] } }
            ],
            "minimum_should_match": 1
          }
        }
      ]
    }
  }
}'

'''
