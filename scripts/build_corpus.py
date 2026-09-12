"""
企业知识库评估语料生成器
========================
生成「近似重复簇」语料：同主题多版本 / 多地区 / 多类型，仅关键数字不同。
目的：让稠密检索产生混淆（多个高度相似文档），从而体现混合检索（稀疏精确
关键词）+ 重排 + Agentic 的价值。

同时生成配套黄金集 golden_set.json，确保 gold_phrases 只出现在目标文档中。

用法：
  python scripts/build_corpus.py            # 生成（覆盖同名文件）
  python scripts/build_corpus.py --clean    # 先清空旧语料目录再生成
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
GOLDEN = ROOT / "SmartQuery" / "evaluation" / "golden_set.json"

_docs = []    # (doc_type, filename, content, dept)
_xlsx = []    # (doc_type, filename, dept, headers, rows)
_golden = []

CN = "一二三四五六七八九十"


def add(doc_type, filename, content, dept="公共"):
    _docs.append((doc_type, filename, content, dept))


def add_xlsx(doc_type, filename, dept, headers, rows):
    _xlsx.append((doc_type, filename, dept, headers, rows))


def q(level, doc_type, question, phrases, reference):
    _golden.append({
        "level": level, "doc_type": doc_type, "question": question,
        "gold_phrases": phrases, "reference": reference,
    })


# ---------------- 文档渲染器（匹配各 doc_type 的分块格式） ----------------

def regulation(title, effective, clauses):
    """规章制度：用「第X条」格式，匹配 _split_by_articles"""
    lines = [f"# {title}", "", f"本制度自{effective}起施行。", ""]
    for i, (t, b) in enumerate(clauses):
        lines += [f"第{CN[i]}条 {t}", b, ""]
    return "\n".join(lines)


def faq(title, pairs):
    """FAQ：Q/A 必须同一行，否则 _split_faq 会在行尾截断导致答案丢失"""
    lines = [f"# {title}", ""]
    for qq, aa in pairs:
        lines += [f"Q：{qq} A：{aa}", ""]
    return "\n".join(lines)


def sop(title, purpose, steps):
    """操作流程SOP：用「步骤X：」格式，匹配 _split_by_steps"""
    lines = [f"# {title}", "", purpose, ""]
    for i, s in enumerate(steps, 1):
        lines += [f"步骤{i}：{s}", ""]
    return "\n".join(lines)


def doc_sections(title, sections):
    """技术文档 / 员工手册：用「##」小节，匹配 _split_by_sections / _split_by_headers"""
    lines = [f"# {title}", ""]
    for h, b in sections:
        lines += [f"## {h}", b, ""]
    return "\n".join(lines)


# ==================== 簇 1：考勤制度多版本 ====================

ATTENDANCE = [
    ("2022", 5, 10, 15, 30, 2),
    ("2023", 6, 11, 16, 40, 3),
    ("2024", 7, 12, 17, 50, 4),
]
for year, a1, a2, a3, late, makeup in ATTENDANCE:
    body = regulation(f"考勤管理制度（{year}版）", f"{year}年1月1日", [
        ("年假", f"工作满1年不满10年的，年休假{a1}天；工作满10年不满20年的，年休假{a2}天；"
                f"工作满20年的，年休假{a3}天。国家法定休假日不计入年休假。"),
        ("迟到", f"迟到10-30分钟，每次扣减绩效工资{late}元；迟到超过30分钟按事假半天处理。"),
        ("补卡", f"每月补卡次数不得超过{makeup}次，超出部分按旷工半天处理。"),
        ("加班", "工作日加班按1.5倍、休息日加班按2倍、法定节假日加班按3倍计发加班费或安排调休。"),
    ])
    add("规章制度", f"考勤管理制度_{year}版.md", body, "HR")
    q("A_事实单跳", "规章制度", f"{year}版考勤制度规定的年假天数是多少？",
      [f"年休假{a1}天", f"年休假{a2}天", f"年休假{a3}天"],
      f"{year}版规定年休假分别为{a1}天、{a2}天、{a3}天。")
    q("E_易混淆", "规章制度", f"{year}版考勤制度中，迟到10到30分钟如何处罚？",
      [f"扣减绩效工资{late}元"], f"{year}版规定迟到10-30分钟扣减绩效工资{late}元。")

# ==================== 簇 2：差旅费标准多地区 ====================

TRAVEL = [
    ("国内一线城市", 600, 150, 200, "北京、上海、广州、深圳"),
    ("国内二线城市", 400, 100, 150, "省会城市及计划单列市"),
    ("海外地区", 1200, 300, 400, "境外及港澳台地区"),
]
for region, hotel, meal, taxi, scope in TRAVEL:
    body = regulation(f"差旅费标准（{region}）", "2024年1月1日", [
        ("住宿标准", f"{region}出差（{scope}）住宿费不超过{hotel}元每晚，超出部分由个人承担。"),
        ("餐饮补贴", f"{region}出差餐补为每天{meal}元，按实际出差天数计发。"),
        ("市内交通", f"{region}出差市内交通费凭票据实报销，每天不超过{taxi}元。"),
    ])
    add("规章制度", f"差旅费标准_{region}.md", body, "财务")
    q("A_事实单跳", "规章制度", f"{region}出差住宿费标准是多少？",
      [f"住宿费不超过{hotel}元"], f"{region}住宿费上限为{hotel}元每晚。")
    q("E_易混淆", "规章制度", f"{region}出差的餐补标准是多少？",
      [f"餐补为每天{meal}元"], f"{region}餐补为每天{meal}元。")

# ==================== 簇 3：请假管理多类型 ====================

LEAVES = [
    ("事假", "每月累计不得超过3天，全年累计不得超过15天", "事假期间不发放工资", "提前1个工作日提交申请"),
    ("病假", "全年累计不得超过30天", "病假期间工资按基本工资的80%发放", "当日提交二级以上医院证明"),
    ("婚假", "婚假为10天，晚婚可增加至15天", "婚假期间工资照常发放", "提前5个工作日提交结婚证复印件"),
    ("产假", "产假为158天，难产可增加15天", "产假期间按规定享受生育津贴", "预产期前1个月提交申请"),
    ("陪产假", "陪产假为15天", "陪产假期间工资照常发放", "配偶分娩后1个月内提交申请"),
    ("丧假", "直系亲属丧假为3天，非直系亲属为1天", "丧假期间工资照常发放", "事发后3个工作日内报备"),
    ("调休假", "调休假应在加班后3个月内使用完毕", "调休期间工资照常发放", "提前1个工作日提交申请"),
    ("工伤假", "工伤假根据医疗机构证明确定，最长不超过24个月", "工伤假期间按规定享受工伤保险待遇", "事故发生后立即报备"),
]
for name, quota, pay, apply_ in LEAVES:
    body = regulation(f"{name}管理规定", "2024年1月1日", [
        ("请假天数", f"{name}规定：{quota}。"),
        ("工资待遇", f"{pay}。"),
        ("申请流程", f"员工申请{name}需{apply_}，经直属主管审批后生效。"),
    ])
    add("规章制度", f"{name}管理规定.md", body, "HR")
    q("A_事实单跳", "规章制度", f"{name}最多可以请多少天？", [quota], f"{name}规定：{quota}。")
    q("E_易混淆", "规章制度", f"{name}期间的工资如何发放？", [pay], f"{name}规定：{pay}。")

# ==================== 簇 4：报销规范多类型 ====================

REIMBURSE = [
    ("差旅费", "单笔不超过50000元", "出差返回后10个工作日内提交", "部门负责人和财务总监"),
    ("招待费", "单次不超过2000元", "消费发生后5个工作日内提交", "部门负责人和分管副总"),
    ("采购报销", "单笔不超过100000元", "收货验收后15个工作日内提交", "采购部、财务总监和总经理"),
    ("培训费", "单次不超过10000元", "培训结束后7个工作日内提交", "人力资源部和部门负责人"),
]
for name, limit, deadline, approver in REIMBURSE:
    body = regulation(f"{name}报销规范", "2024年1月1日", [
        ("报销限额", f"{name}{limit}，超出限额需提前专项审批。"),
        ("报销时限", f"{name}应在{deadline}，逾期需书面说明原因。"),
        ("审批流程", f"{name}需经{approver}审批后交财务付款。"),
    ])
    add("规章制度", f"{name}报销规范.md", body, "财务")
    q("A_事实单跳", "规章制度", f"{name}报销的时限要求是多久？", [deadline], f"{name}应在{deadline}。")
    q("E_易混淆", "规章制度", f"{name}的报销金额上限是多少？", [f"{name}{limit}"], f"{name}{limit}。")

# ==================== 簇 5：巡检SOP ====================

INSPECT = [
    ("服务器巡检", "每日一次", "CPU、内存、磁盘使用率", "CPU使用率超过80%"),
    ("数据库巡检", "每日两次", "连接数、慢查询、主从延迟", "主从延迟超过10秒"),
    ("网络设备巡检", "每周一次", "端口状态、带宽利用率、丢包率", "丢包率超过1%"),
    ("备份恢复巡检", "每月一次", "备份成功率、恢复演练结果", "备份失败"),
]
for name, interval, items, threshold in INSPECT:
    body = sop(f"{name}SOP", f"本SOP用于规范{name}的执行标准。", [
        f"确认巡检范围，包括{items}。",
        f"执行巡检，{name}{interval}执行一次，逐项记录指标。",
        f"告警处理，当{threshold}时立即上报值班工程师。",
        "归档，巡检记录保存不少于6个月。",
    ])
    add("操作流程SOP", f"{name}SOP.md", body, "IT")
    q("A_事实单跳", "操作流程SOP", f"{name}多久执行一次？",
      [f"{name}{interval}执行一次"], f"{name}{interval}执行一次。")
    q("E_易混淆", "操作流程SOP", f"{name}的告警阈值是什么？",
      [f"当{threshold}时"], f"{name}在{threshold}时告警。")

# ==================== 簇 6：微服务API ====================

APIS = [
    ("用户中心", "/api/v1/user", 1000, "OAuth2.0", "3秒"),
    ("订单中心", "/api/v1/order", 2000, "JWT", "5秒"),
    ("支付中心", "/api/v1/pay", 500, "HMAC签名", "10秒"),
    ("库存中心", "/api/v1/stock", 1500, "JWT", "2秒"),
    ("消息中心", "/api/v1/message", 3000, "API Key", "1秒"),
    ("商品中心", "/api/v1/product", 2500, "JWT", "4秒"),
    ("会员中心", "/api/v1/member", 1800, "OAuth2.0", "3秒"),
    ("营销中心", "/api/v1/promotion", 900, "API Key", "6秒"),
    ("物流中心", "/api/v1/logistics", 1200, "HMAC签名", "8秒"),
    ("搜索中心", "/api/v1/search", 4000, "API Key", "2秒"),
    ("评价中心", "/api/v1/review", 700, "JWT", "3秒"),
    ("风控中心", "/api/v1/risk", 600, "HMAC签名", "5秒"),
]
for name, path, qps, auth, timeout in APIS:
    body = doc_sections(f"{name}API接入文档", [
        ("接口地址", f"{name}的接口地址为{path}，仅限内网调用。"),
        ("鉴权方式", f"{name}采用{auth}鉴权，需在请求头携带凭证。"),
        ("限流策略", f"{name}接口限流为{qps} QPS，超出部分返回429状态码。"),
        ("超时设置", f"{name}接口调用超时时间为{timeout}，建议客户端设置重试。"),
    ])
    add("技术文档", f"{name}API接入文档.md", body, "技术")
    q("A_事实单跳", "技术文档", f"{name}API的限流是多少QPS？",
      [f"限流为{qps} QPS"], f"{name}限流{qps} QPS。")
    q("E_易混淆", "技术文档", f"{name}API采用什么鉴权方式？",
      [f"采用{auth}鉴权"], f"{name}采用{auth}鉴权。")

# ==================== 簇 7：信息安全规范 ====================

INFOSEC = [
    ("密码策略", "密码长度不少于12位，需包含大小写字母、数字和特殊字符中的至少三类",
     "普通系统每90天更换一次，核心系统每60天更换一次"),
    ("数据分级", "数据分为公开、内部、秘密、机密四个级别",
     "机密级数据每季度复核一次"),
    ("权限申请", "权限申请遵循最小权限原则，需直属主管审批",
     "权限每半年复核一次"),
    ("数据外发", "秘密级以上数据禁止外发，外发需加密并留存记录",
     "外发记录保留3年"),
]
for name, rule, cycle in INFOSEC:
    body = regulation(f"{name}规范", "2024年1月1日", [
        ("基本要求", f"{name}的基本要求是：{rule}。"),
        ("复核周期", f"{name}的复核要求为：{cycle}。"),
        ("违规处理", f"违反{name}的，视情节给予警告直至解除劳动合同处理。"),
    ])
    add("规章制度", f"{name}规范.md", body, "IT")
    q("A_事实单跳", "规章制度", f"{name}的基本要求是什么？", [rule], f"{name}：{rule}。")
    q("E_易混淆", "规章制度", f"{name}的复核周期是多久？", [cycle], f"{name}复核：{cycle}。")

# ==================== 簇 8：员工手册多主体 ====================

HANDBOOKS = [
    ("总部", "9:00-18:00", "10天", "每月600元"),
    ("华东分公司", "9:30-18:30", "12天", "每月800元"),
    ("华南分公司", "8:30-17:30", "11天", "每月700元"),
]
for name, hours, annual, meal in HANDBOOKS:
    body = doc_sections(f"{name}员工手册", [
        ("工作时间", f"{name}实行标准工时制，工作时间为{hours}，中午休息1小时。"),
        ("带薪年假", f"{name}员工入职满一年后享受带薪年假{annual}。"),
        ("福利补贴", f"{name}提供{meal}餐补，随当月工资发放。"),
        ("行为规范", "员工应遵守公司各项规章制度，维护公司形象，不得从事损害公司利益的行为。"),
    ])
    add("员工手册", f"员工手册_{name}.md", body, "公共")
    q("A_事实单跳", "员工手册", f"{name}员工手册规定的工作时间是什么？",
      [f"工作时间为{hours}"], f"{name}工作时间为{hours}。")
    q("E_易混淆", "员工手册", f"{name}的餐补标准是多少？", [f"{meal}餐补"], f"{name}餐补为{meal}。")

# ==================== 簇 9：FAQ ====================

FAQ_DOCS = [
    ("HR常见问题", "HR", [
        ("如何办理入职手续？", "携带身份证、学历证明、离职证明到人力资源部办理入职手续。"),
        ("试用期一般多久？", "普通岗位试用期为3个月，管理岗位试用期为6个月。"),
        ("工资什么时候发放？", "工资于每月15日发放，遇节假日提前至最近工作日。"),
    ]),
    ("IT常见问题", "IT", [
        ("如何申请VPN账号？", "在IT服务台提交申请，经部门负责人审批后由IT开通。"),
        ("电脑故障如何报修？", "在IT服务台提交工单，一般4小时内响应。"),
        ("公司无线网络怎么连接？", "公司无线网络使用个人账号密码登录，密码与域账号一致。"),
    ]),
    ("财务常见问题", "财务", [
        ("发票丢失如何处理？", "需提供发票复印件及情况说明，经财务审核后方可报销。"),
        ("报销多久到账？", "报销审核通过后3到5个工作日到账。"),
        ("如何查询个人报销进度？", "登录财务系统在报销查询模块查看进度。"),
    ]),
    ("行政常见问题", "公共", [
        ("如何预订会议室？", "通过办公系统预订会议室，需提前1个工作日预约。"),
        ("办公用品如何领用？", "在行政系统提交领用申请，审批后到前台领取。"),
        ("快递如何寄送？", "公司快递由前台统一寄送，个人快递费用自理。"),
    ]),
    ("新员工常见问题", "HR", [
        ("入职需要准备哪些材料？", "身份证、学历证书、离职证明、体检报告和一寸照片。"),
        ("入职第一天需要做什么？", "到人力资源部报到，领取工牌和办公用品，参加入职培训。"),
        ("试用期可以请假吗？", "试用期内可以请假，但累计请假超过5天将顺延试用期。"),
    ]),
]
for title, dept, pairs in FAQ_DOCS:
    add("FAQ", f"{title}.md", faq(title, pairs), dept)
    for qq, aa in pairs:
        q("A_事实单跳", "FAQ", qq, [aa], aa)

# ==================== 簇 10：系统架构 ====================

ARCHS = [
    ("电商中台", "Spring Cloud、MySQL、Redis", "5万QPS", "Kubernetes多可用区部署"),
    ("数据平台", "Flink、Kafka、ClickHouse", "日均处理10亿条数据", "Hadoop集群部署"),
    ("风控系统", "Python、Flink、Redis", "2万QPS", "同城双活部署"),
    ("推荐系统", "Python、TensorFlow、Redis", "3万QPS", "GPU集群部署"),
    ("支付网关", "Java、MySQL、Kafka", "1万QPS", "两地三中心部署"),
    ("消息推送", "Go、Kafka、Redis", "日均推送5亿条", "多机房部署"),
    ("搜索服务", "Java、Elasticsearch、Redis", "4万QPS", "多可用区部署"),
    ("用户画像", "Spark、HBase、Kafka", "日均更新1亿用户", "Hadoop集群部署"),
    ("监控平台", "Prometheus、Grafana、VictoriaMetrics", "每秒采集100万指标", "多可用区部署"),
    ("配置中心", "Java、MySQL、Nacos", "1万QPS", "三机房部署"),
    ("网关服务", "Go、Redis、Etcd", "8万QPS", "多可用区部署"),
    ("日志平台", "Elasticsearch、Logstash、Kibana", "日均写入10TB", "冷热分层部署"),
]
for name, stack, qps, deploy in ARCHS:
    body = doc_sections(f"{name}系统架构说明", [
        ("技术栈", f"{name}采用{stack}构建，各组件通过内网通信。"),
        ("性能指标", f"{name}的峰值处理能力为{qps}。"),
        ("部署架构", f"{name}采用{deploy}，保障高可用。"),
        ("监控告警", f"{name}接入统一监控平台，核心指标异常时自动告警。"),
    ])
    add("技术文档", f"{name}系统架构说明.md", body, "技术")
    q("A_事实单跳", "技术文档", f"{name}的技术栈是什么？", [stack], f"{name}采用{stack}。")
    q("A_事实单跳", "技术文档", f"{name}的峰值处理能力是多少？", [qps], f"{name}峰值{qps}。")

# ==================== 簇 10b：数据库表结构（字段型近似文档） ====================

TABLES = [
    ("t_user", "用户表", "user_id", "username", "idx_username"),
    ("t_order", "订单表", "order_id", "order_no", "idx_order_no"),
    ("t_payment", "支付表", "payment_id", "trade_no", "idx_trade_no"),
    ("t_product", "商品表", "product_id", "sku_code", "idx_sku"),
    ("t_stock", "库存表", "stock_id", "warehouse_code", "idx_warehouse"),
    ("t_member", "会员表", "member_id", "member_card", "idx_card"),
    ("t_coupon", "优惠券表", "coupon_id", "coupon_code", "idx_coupon"),
    ("t_logistics", "物流表", "logistics_id", "waybill_no", "idx_waybill"),
    ("t_review", "评价表", "review_id", "review_no", "idx_review"),
    ("t_risk", "风控表", "risk_id", "event_no", "idx_event"),
    ("t_message", "消息表", "message_id", "msg_no", "idx_msg"),
    ("t_search", "搜索表", "search_id", "keyword", "idx_keyword"),
]
for table, name, pk, field, idx in TABLES:
    body = doc_sections(f"{name} {table} 表结构说明", [
        ("主键", f"{table} 表的主键为 {pk}，采用自增策略。"),
        ("关键字段", f"{table} 表的关键字段为 {field}，业务上要求唯一。"),
        ("索引", f"{table} 表在 {idx} 上建立了索引，用于加速查询。"),
        ("分库分表", f"{table} 表按 {pk} 哈希分 16 张表。"),
    ])
    add("技术文档", f"表结构_{table}.md", body, "技术")
    q("E_易混淆", "技术文档", f"{table} 表的主键是什么？",
      [f"主键为 {pk}"], f"{table} 主键为 {pk}。")
    q("E_易混淆", "技术文档", f"{table} 表建立了什么索引？",
      [f"在 {idx} 上建立了索引"], f"{table} 在 {idx} 上建索引。")

# ==================== 簇 11：数据报表（xlsx） ====================

add_xlsx("数据报表", "差旅费标准表.xlsx", "财务",
         ["城市类别", "住宿上限元每晚", "餐补元每天", "市内交通元每天"],
         [["国内一线城市", 600, 150, 200],
          ["国内二线城市", 400, 100, 150],
          ["海外地区", 1200, 300, 400]])
add_xlsx("数据报表", "职级薪酬表.xlsx", "HR",
         ["职级", "月薪下限", "月薪上限", "年终奖月数"],
         [["P4", 15000, 25000, 2],
          ["P5", 25000, 40000, 3],
          ["P6", 40000, 60000, 4]])
add_xlsx("数据报表", "考勤统计表.xlsx", "HR",
         ["部门", "出勤率", "平均加班小时", "异常次数"],
         [["研发部", "98%", 12, 3],
          ["市场部", "96%", 8, 5],
          ["财务部", "99%", 4, 1]])
add_xlsx("数据报表", "部门预算表.xlsx", "财务",
         ["部门", "年度预算万元", "已使用万元", "使用率"],
         [["研发部", 2000, 1200, "60%"],
          ["市场部", 1500, 900, "60%"],
          ["人力资源部", 800, 400, "50%"]])

q("G_结构化查询", "数据报表", "P5职级的月薪范围是多少？",
  ["25000", "40000"], "P5职级月薪为25000至40000元。")
q("G_结构化查询", "数据报表", "海外地区在差旅费标准表中的住宿上限是多少？",
  ["1200"], "差旅费标准表中海外地区住宿上限为1200元每晚。")

# ==================== 难度分层补充：B 术语理解 ====================

TERMS = [
    ("SLA", "SLA 即服务等级协议（Service Level Agreement），指服务方对可用性、响应时间等指标的承诺。"),
    ("QPS", "QPS 即每秒查询数（Queries Per Second），用于衡量系统吞吐量。"),
    ("RTO", "RTO 即恢复时间目标（Recovery Time Objective），指故障后恢复业务的最长可接受时间。"),
    ("RPO", "RPO 即恢复点目标（Recovery Point Objective），指可接受的最大数据丢失时间窗口。"),
    ("RRF", "RRF 即倒数排名融合（Reciprocal Rank Fusion），一种多路检索结果的融合算法。"),
    ("HyDE", "HyDE 即假设文档嵌入（Hypothetical Document Embeddings），先生成假设答案再检索的方法。"),
    ("倒排索引", "倒排索引是一种从词项映射到文档列表的索引结构，用于加速关键词检索。"),
    ("向量量化", "向量量化是一种压缩向量、降低内存占用的技术，常见方法有乘积量化 PQ 和标量量化 SQ。"),
]
add("技术文档", "技术术语说明.md", doc_sections("技术术语说明", [(t, d) for t, d in TERMS]), "技术")
for t, d in TERMS:
    q("B_术语理解", "技术文档", f"{t} 是什么意思？", [d], d)

# ==================== 难度分层补充：C 语义改写 ====================

PARAPHRASE = [
    ("规章制度", "我想请事假，一个月最多能请几天？", ["每月累计不得超过3天"], "事假每月累计不得超过3天。"),
    ("规章制度", "病假一年最多能休多久？", ["全年累计不得超过30天"], "病假全年累计不得超过30天。"),
    ("规章制度", "公司报销差旅费有时间要求吗？", ["出差返回后10个工作日内提交"], "差旅费应在出差返回后10个工作日内提交。"),
    ("规章制度", "系统密码多久需要换一次？", ["普通系统每90天更换一次", "核心系统每60天更换一次"], "普通系统90天、核心系统60天更换一次。"),
    ("规章制度", "员工一年能休多少天带薪年假？", ["年休假7天", "年休假12天", "年休假17天"], "现行考勤制度规定年休假7/12/17天。"),
    ("技术文档", "用户中心接口每秒最多能调用多少次？", ["限流为1000 QPS"], "用户中心接口限流1000 QPS。"),
    ("技术文档", "订单中心接口怎么鉴权？", ["采用JWT鉴权"], "订单中心采用JWT鉴权。"),
    ("操作流程SOP", "服务器多久检查一次？", ["服务器巡检每日一次执行一次"], "服务器巡检每日一次。"),
    ("员工手册", "总部上班时间是几点到几点？", ["工作时间为9:00-18:00"], "总部工作时间为9:00-18:00。"),
]
for dt, question, phrases, ref in PARAPHRASE:
    q("C_语义改写", dt, question, phrases, ref)

# ==================== 难度分层补充：D 多跳推理 ====================

MULTIHOP = [
    ("规章制度", "海外出差住宿标准比国内一线城市高多少元？",
     ["住宿费不超过1200元", "住宿费不超过600元"], "海外1200元、国内一线600元，相差600元。"),
    ("规章制度", "国内一线和二线城市住宿标准相差多少元？",
     ["住宿费不超过600元", "住宿费不超过400元"], "国内一线600元、二线400元，相差200元。"),
    ("规章制度", "产假和陪产假一共多少天？",
     ["产假为158天", "陪产假为15天"], "158+15=173天。"),
    ("技术文档", "用户中心和订单中心接口限流相差多少QPS？",
     ["限流为1000 QPS", "限流为2000 QPS"], "用户中心1000、订单中心2000，相差1000 QPS。"),
]
for dt, question, phrases, ref in MULTIHOP:
    q("D_多跳推理", dt, question, phrases, ref)

# ==================== 难度分层补充：F 跨文档 ====================

CROSS = [
    ("规章制度", "国内一线、国内二线和海外地区的住宿标准分别是多少？",
     ["住宿费不超过600元", "住宿费不超过400元", "住宿费不超过1200元"], "分别为600/400/1200元。"),
    ("规章制度", "事假和病假的年度上限分别是多少？",
     ["全年累计不得超过15天", "全年累计不得超过30天"], "事假15天、病假30天。"),
    ("技术文档", "用户中心、订单中心、支付中心的接口限流分别是多少？",
     ["限流为1000 QPS", "限流为2000 QPS", "限流为500 QPS"], "分别为1000/2000/500 QPS。"),
]
for dt, question, phrases, ref in CROSS:
    q("F_跨文档", dt, question, phrases, ref)

# ==================== 单例文档 ====================

add("操作流程SOP", "应急响应流程.md", sop("应急响应流程", "本流程用于规范生产环境故障的应急处理。", [
    "故障发现，值班人员通过监控告警或用户反馈发现故障。",
    "故障定级，P1为业务中断，P2为部分功能不可用，P3为体验问题。",
    "上报通知，P1故障5分钟内通知技术负责人和业务负责人。",
    "止损处理，优先恢复业务，可通过回滚、限流、降级等手段。",
    "复盘归档，故障恢复后3个工作日内输出复盘报告。",
]), "IT")

add("技术文档", "系统架构说明.md", doc_sections("系统架构说明", [
    ("整体架构", "系统采用前后端分离架构，后端基于FastAPI提供REST接口，前端使用Vue构建。"),
    ("数据存储", "结构化数据存储于MySQL，向量数据存储于Milvus，缓存使用Redis。"),
    ("模型服务", "大模型通过OpenAI兼容接口调用，嵌入模型用于向量化。"),
    ("部署方式", "服务以Docker容器化部署，通过Nginx反向代理对外提供服务。"),
]), "技术")

add("员工手册", "员工手册.md", doc_sections("员工手册", [
    ("企业文化", "公司倡导诚信、协作、创新的价值观，鼓励员工持续学习与成长。"),
    ("考勤制度", "员工应按时上下班，具体考勤规则以考勤管理制度为准。"),
    ("薪酬福利", "公司按月发放工资，提供五险一金、带薪年假和节日福利。"),
    ("职业发展", "公司提供管理与专业双通道晋升路径，定期开展绩效评估。"),
]), "公共")


def write_xlsx(path, headers, rows):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = path.stem
    ws.append(headers)
    for r in rows:
        ws.append(r)
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description="生成企业知识库评估语料 + 黄金集")
    ap.add_argument("--clean", action="store_true", help="生成前清空语料目录")
    args = ap.parse_args()

    if args.clean and CORPUS.exists():
        shutil.rmtree(CORPUS)
        print(f"已清空 {CORPUS}")

    dept_map = {}
    for doc_type, filename, content, dept in _docs:
        d = CORPUS / doc_type
        d.mkdir(parents=True, exist_ok=True)
        (d / filename).write_text(content, encoding="utf-8")
        dept_map[filename] = dept

    for doc_type, filename, dept, headers, rows in _xlsx:
        d = CORPUS / doc_type
        d.mkdir(parents=True, exist_ok=True)
        write_xlsx(d / filename, headers, rows)
        dept_map[filename] = dept

    (CORPUS / "_departments.json").write_text(
        json.dumps(dept_map, ensure_ascii=False, indent=2), encoding="utf-8")
    GOLDEN.write_text(
        json.dumps(_golden, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"生成完成：文档 {len(_docs) + len(_xlsx)} 篇，黄金问题 {len(_golden)} 条")
    print(f"语料目录：{CORPUS}")
    print(f"黄金集：{GOLDEN}")


if __name__ == "__main__":
    main()
