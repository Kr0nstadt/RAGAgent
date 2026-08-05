#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the agent_process_plan.json for the bakery MTO process.

Sources:
- agent_context.json (12 process blocks)
- answers.json (60 user-confirmed decisions)
- process_plan.json mappings (graph-only, populated earlier by plan-task)
- dependencies (ЗаказКлиента, ПланПроизводства, ЗаказПоставщику, ВозвратТоваровОтКлиента,
  ПриобретениеТоваровУслуг, ЗаказНаПроизводство2_2, ЭтапПроизводства2_2,
  РеализацияТоваровУслуг, ПеремещениеТоваров, ОтборРазмещениеТоваров,
  АктКонтроляКачестваТоваров, ЗаказНаДоставку)
- document-chain (ЗаказКлиента→Реализация, ЗаказПоставщику→Приобретение,
  ЗаказНаПроизводство2_2→ЭтапПроизводства2_2, Реализация→Возврат,
  АктКонтроляКачестваТоваров→Приобретение)
- query_graph.py (Заказ на доставку semantics, 5.3 Формирование заказов по потребностям)

Evidence rules:
- Every metadata name has graph node_id (ERPcode/Documents/... or ERPcode/Catalogs/...)
  or scenario_* ID for L1 scenarios.
- ui_paths only when document-chain/dependencies provided them.
- register/registrator pairs only when from dependencies/chain output.
- question IDs from answers.json referenced as confirmed-decision (not guess).
"""

import json
import os

BASE = "C:/Users/Y.Karpova/Desktop/RAGAgent/task_data/подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f"

# ------------------------------------------------------------------
# Constants from confirmed answers (answers.json + assumptions)
# ------------------------------------------------------------------

# Confirmed decisions (key question IDs from answers.json)
ANS = {
    "q03_roles":                        "подтверждаю предложенный список",
    "q05_own_store":                    "внутреннее перемещение (отдельный склад)",
    "q_own_store_isolated":             "аудит q05 — собственный магазин = отдельный склад (внутр. перемещение)",
    "audit_q01_treasury":               "нужно (оперативный + управленческий по денежным средствам, без бухгалтерского)",
    "audit_q02_pricing":                "единый прайс + соглашения для опт/сеть/собств. магазин + внутр. себестоимостная цена для собственного магазина",
    "audit_q03_roles":                  "подтверждаю предложенный список ролей",
    "audit_q04_acceptance":             "контрольные примеры позже",
    "serii_hleb_5_days":                "нужно, сроки только на хлеб 5 дней",
    "mto_only":                         "только под заказ",
    "quality_control_owner":            "кладовщик при приеме товара",
    "manager_confirmation":             "подтверждение менеджера",
    "price_average_purchase":           "по средней закупочной",
    "polufabrikat_separate_nomenclature":"a — отдельная номенклатура и отдельный выпуск",
    "acceptance_full_4states":          "все: принять/принять частично/изолировать/вернуть/списать; частичная приемка через возврат поставщику",
    "serii_systemic":                   "b — системно по сериям (срок годности)",
    "vozvrat_dengi":                    "возврат денег",
    "ordernaya_schema":                 "да, ордерная схема везде",
    "rashod_avg_N_days":                "a — средний расход за N дней",
    "different_prices_per_counterparty":"цены для каждого контрагента свои; доставку опт/сеть делают сами; в свой магазин — мы сами, внутреннее перемещение на своем транспорте",
    "min_zapas_pos_sk_period":          "да, минимальный запас по позиции+складу+периоду для расходников и материалов (не для готовой продукции)",
    "units_only_shtuki":                "только штуки",
    "all_sklsady_confirmed":            "склад сырья, кладовая холодного цеха, кладовая горячего цеха, склад ГП, склад собственного магазина, склад возврата/утилизации",
    "zakaz_na_proizv_each":             "a — Заказ на производство по каждому заказу клиента (на 1–2 дня)",
    "samovyvoz":                        "b — ничего, клиент сам приезжает",
    "etap_22_two_podrazdeleniya":       "a — Этап производства 2.2 с двумя подразделениями (холодный/горячий) b — Перемещение товаров между складами. Перемещение между цехами нужно",
    "edit_nakladnaya":                  "редактирование накладной",
    "priemka_6_3":                      "a — 6.3. Приемка товаров на склад",
    "spisanie_needed":                  "нужно списание",
    "hleb_units_upakovki":              "да — хлебобулочные изделия в штуках, упаковки через справочник Упаковки (не отдельная номклатура)",
    "finish_in_gorach":                 "в горячем цеху приготовление продукции завершается — Этап производства 2.2 (подразделение = горячий цех)",
    "etap22_perem":                     "да — ЭтапПроизводства2_2 + ПеремещениеТоваров",
    "sklad_gp_otborrazm":               "да — склад ГП через ОтборРазмещениеТоваров → РеализацияТоваровУслуг",
    "tovary_na_skl":                    "да — регистр ТоварыНаСкладах, отчёт «Остатки товаров на складах»",
    "otchet_dvizeniy":                  "отчёт по движениям",
    "zakaz_na_dostavku_sob":            "используется ЗаказНаДоставку для собственного магазина",
    "kontrol_1c_erp":                   "b — контроль в 1С:ERP",
    "samovyvoz_flag":                   "да — флаг «Самовывоз» / способ доставки в Заказе клиента",
    "autozapoln_zakazov":               "a — автозаполнение ЗаказовНаПроизводство",
    "res_spec":                         "да — РесурсныеСпецификации (полуфабрикат Тесто)",
    "fifo_policy":                      "a — ФИФО в учётной политике",
    "tovary_nask_shtuki":               "да — ТоварыНаСкладах (хранение остатков ГП в шт.)",
    "perem_tov_yes":                    "да — ПеремещениеТоваров",
    "res_spec_yes":                     "да — ресурсные спецификации",
    "zarplata_off":                     "да — ИспользоватьЗарплатаИКадры = Ложь, зарплата во внешнем приложении",
    "kontrol_serii":                    "a — контроль через серии (отчёт)",
    "mnoogooborotnaya_ne_using":        "многооборотная тара не используется",
    "vozvrat_tov_ot_klienta":           "да — ВозвратТоваровОтКлиента на основании РеализацияТоваровУслуг",
    "form_zakazov_5_3":                 "a — 5.3 Формирование заказов по потребностям",
    "podsystema_proizv":                "a — Подсистема Производство",
    "dannye_zakazklient":               "да — данные через ЗаказКлиента (статус «К производству»)",
    "otchet_sebestoimost":              "a — отчёт «Себестоимость товаров»",
    "res_spec_yes2":                    "a — ресурсные спецификации",
    "grafik_dostavki":                  "да — ГрафикДоставки / СрокОтгрузки",
    "etap22_two_podr":                  "да — ЭтапПроизводства2_2 с подразделениями в ЗаказНаПроизводство",
    "vypusk_cherez_etap":               "да — выпуск через этап производства, приход на кладовую полуфабрикатов",
    "priobr_tov_uslug":                 "да — ПриобретениеТоваровУслуг",
    "realiz_torg12_transportnaya":      "да — РеализацияТоваровУслуг + ТОРГ-12 + ТранспортнаяНакладная",
    "uchet_po_partiyam":                "да — учёт по партиям включён",
    "akt_kontrolya_kach":               "a — АктКонтроляКачестваТоваров",
    "auto_proverka_nastroyka":          "a — автоматическая проверка через настройку",
}

# Graph evidence IDs
NSI = {
    "partners":     "ERPcode/Catalogs/Партнеры",
    "kontragenty":  "ERPcode/Catalogs/Контрагенты",
    "orgs":         "ERPcode/Catalogs/Организации",
    "dogovory":     "ERPcode/Catalogs/ДоговорыКонтрагентов",
    "soglasiya":    "ERPcode/Catalogs/Соглашения об условиях продаж",
    "valuty":       "ERPcode/Catalogs/Валюты",
    "valuty_vzaim": "ERPcode/Catalogs/Валюты (ВалютаРегламентированногоУчета / ВалютаВзаиморасчетов)",
    "struktura":    "ERPcode/Catalogs/Структура предприятия",
    "prioritety":   "ERPcode/Catalogs/Приоритеты",
    "nomenklatura": "ERPcode/Catalogs/Номенклатура",
    "upakovki":     "ERPcode/Catalogs/Упаковки",
    "harakteristiki": "ERPcode/Catalogs/ХарактеристикиНоменклатуры",  # not strictly in graph - see gaps
    "kladovye":     "ERPcode/Catalogs/Склады",
    "partii_proizv":"ERPcode/InformationRegisters/ПартииПроизводства",
    "res_spec":     "ERPcode/Catalogs/РесурсныеСпецификации",
    "stages_res":   "ERPcode/Catalogs/ЭтапыПроизводства",
    "vidceny":      "ERPcode/Catalogs/ВидыЦен",
    "ceny_post":    "ERPcode/InformationRegisters/ЦеныНоменклатурыПоставщиков",
    "ceny_prod":    "ERPcode/InformationRegisters/ЦеныНоменклатуры",  # covered by dep_zakaz_clienta
    "sklady":       "ERPcode/Catalogs/Склады",
    "kassa":        "ERPcode/Documents/ПриходныйКассовыйОрдер",
    "vygruzka":     "ERPcode/Documents/РасходныйКассовыйОрдер",      # placeholder - see gaps
    "plan_prod":    "ERPcode/Documents/ПланПроизводства",
    "zakaz_kl":     "ERPcode/Documents/ЗаказКлиента",
    "zakaz_post":   "ERPcode/Documents/ЗаказПоставщику",
    "priobr":       "ERPcode/Documents/ПриобретениеТоваровУслуг",
    "vozvrat_post": "ERPcode/Documents/ВозвратТоваровПоставщику",
    "akk":          "ERPcode/Documents/АктКонтроляКачестваТоваров",
    "zakaz_proizv":"ERPcode/Documents/ЗаказНаПроизводство2_2",
    "etap_proizv": "ERPcode/Documents/ЭтапПроизводства2_2",
    "vypusk":       "ERPcode/Documents/ВыпускПродукции",
    "perem":        "ERPcode/Documents/ПеремещениеТоваров",
    "otbor_razm":   "ERPcode/Documents/ОтборРазмещениеТоваров",
    "realiz":       "ERPcode/Documents/РеализацияТоваровУслуг",
    "vozvrat_kl":   "ERPcode/Documents/ВозвратТоваровОтКлиента",
    "zakaz_dost":   "ERPcode/Documents/ЗаказНаДоставку",
    "rasporyazh":   "ERPcode/Documents/РаспоряжениеНаОтгрузку",  # placeholder
    "is_mp":        "ERPcode/Documents/РазрешениеНаОтгрузкуИСМП", # may be a gap
    "akt_brak":     "ERPcode/Documents/АктОСписанииТоваровУТМ",  # placeholders
    "spis_tov":     "ERPcode/Documents/СписаниеТоваров",
    "tovary_nask":  "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
    "tovary_otgr":  "ERPcode/AccumulationRegisters/ТоварыКОтгрузке",
    "rasporyazh_otgr_vozvr": "ERPcode/AccumulationRegisters/РаспоряженияНаОтгрузкуИВозврат",
    "raschety_kl":  "ERPcode/AccumulationRegisters/РасчетыСКлиентами",
    "raschety_post":"ERPcode/AccumulationRegisters/РасчетыСПоставщиками",
    "rasp_post_plan": "ERPcode/AccumulationRegisters/РасчетыСПоставщикамиПланПоставок",
    "rasp_kl_plan_otgr": "ERPcode/AccumulationRegisters/РасчетыСКлиентамиПланОтгрузок",
    "rasp_kl_plan_oplat":"ERPcode/AccumulationRegisters/РасчетыСКлиентамиПланОплат",
    "plan_vypuska": "ERPcode/AccumulationRegisters/ПланыВыпускаИзделий",
    "dvizh_seriy":  "ERPcode/AccumulationRegisters/ДвиженияСерийТоваров",
    "dostavka":     "ERPcode/AccumulationRegisters/Доставка",
    "zapasy_potr":  "ERPcode/AccumulationRegisters/ЗапасыИПотребности",
    "ostat_tmc":    "ERPcode/AccumulationRegisters/ОстаткиТМЦ",  # placeholder
    "par_post":     "ERPcode/AccumulationRegisters/ПартииТоваровОрганизаций",
    "report_ostat_tov":   "ERPcode/Reports/ОстаткиТоваровНаСкладах",
    "report_sebest":      "ERPcode/Reports/СебестоимостьТоваров",
    "report_dvizhenia":   "ERPcode/Reports/ДвиженияТоваров",  # placeholder (need verify)
    "report_kontr_dok":   "ERPcode/Reports/КонтрольОформленияДокументовТовародвижений",
    "report_sebest_pr":   "ERPcode/Reports/ПлановаяСебестоимостьПродукции",
    "report_fakt_sebest": "ERPcode/Reports/ФактическаяСебестоимостьПродукции",
    "report_plan_vyp":    "ERPcode/Reports/ПланВыпускаИзделий",  # placeholder
    "scenario_5_3":  "scenario_003--5.3. Формирование заказов по потребностям",
    "scenario_2_10": "scenario_010--2.10. Планирование производства",
    "scenario_2_12": "scenario_012--2.12. Планирование закупок",
    "scenario_6_3":  "scenario_003--6.3. Приемка товаров на склад",
    "scenario_6_4":  "scenario_004--6.4. Контроль качества товаров",
    "scenario_6_5":  "scenario_005--6.5. Отгрузка товаров",
    "scenario_9":    "scenario_010--9. Производство",
    "scenario_9_2":  "scenario_010--9.2. Производство 2.2",
    "scenario_14_6": "scenario_006--14.6. Особенности методологии учета",
    "scenario_2_18": "scenario_018--2.18. Формирование заказов поставщикам по планам",
    "scenario_2_7":  "scenario_007--2.7. Планирование остатков",
    "scenario_5_5":  "scenario_005--5.5. Распределение запасов",
    "scenario_5_6":  "scenario_006--5.6. Состояние обеспечения заказов",
    "scenario_7_15": "scenario_015--7.15. Прием на ответственное хранение",
    "scenario_7_20": "scenario_020--7.20. Заявки на закупку и обеспечение",
}

# ------------------------------------------------------------------
# Plan structure
# ------------------------------------------------------------------
process_blocks = []

def block(num, title, source_nodes, scenario_id=None, scenario_title=None,
          scenario_layer=None, summary=None, allowed_with_blocked=None):
    """Construct a process block descriptor.

    allowed_with_blocked: list of gap question IDs still open for this block
    """
    return {
        "block_id": num,
        "title": title,
        "source_block_id": source_nodes[0] if source_nodes else None,
        "scenario": ({"id": scenario_id, "title": scenario_title, "layer": scenario_layer}
                    if scenario_id else None),
        "summary": summary,
        "open_gaps": allowed_with_blocked or [],
    }

# -----------------------------------------------------------------
# 12 BP blocks — text source from task_graph.json, ER mapping from
# process_plan.json/normalization/normalization gaps.
# -----------------------------------------------------------------

block1 = {
    "block_id": 1,
    "title": "ПРИЕМ И ОБРАБОТКА ЗАКАЗОВ",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:1",
    "scenario": None,
    "summary": (
        "Клиент (собственный магазин, опт, сетевая розница) подает заявку на отгрузку "
        "хлебобулочной продукции до 12:00 текущего дня. Заявка фиксируется как "
        "ЗаказКлиента со статусом 'К производству' и параметром Самовывоз/Доставка."
    ),
    "open_gaps": ["question:gap:12b2d529a863893f"],
    "steps": [
        {
            "step_id": "1.1",
            "name": "Регистрация заявки в системе",
            "event": "Клиент подает заявку на отгрузку продукции до 12:00",
            "primary_document": {
                "node_id": "ERPcode/Documents/ЗаказКлиента",
                "title": "Документ Заказ клиента",
                "ui_paths": [
                    "Продажи → ОптовыеПродажи → ЗаказКлиента",
                    "ПродажиБазовая → ВедениеЗаказовКлиентовБазовая → ЗаказКлиента",
                ],
                "evidence": "process_plan.json operational_steps[0].ui_paths + chain_klient_realiz.json",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Партнеры",
                 "create_from_form": True,
                 "evidence": "dep_zakaz_clienta.json (required-by-metadata, inline-candidate)"},
                {"node_id": "ERPcode/Catalogs/Контрагенты",
                 "create_from_form": True,
                 "evidence": "dep_zakaz_clienta.json (required-by-metadata, inline-candidate)"},
                {"node_id": "ERPcode/Catalogs/Соглашения об условиях продаж",
                 "create_from_form": True,
                 "evidence": "dep_zakaz_clienta.json (required-by-metadata, inline-candidate)"},
                {"node_id": "ERPcode/Catalogs/ДоговорыКонтрагентов",
                 "create_from_form": True,
                 "evidence": "dep_zakaz_clienta.json (required-by-metadata, inline-candidate)"},
                {"node_id": "ERPcode/Catalogs/Организации",
                 "create_from_form": False,
                 "evidence": "dep_zakaz_clienta.json"},
                {"node_id": "ERPcode/Catalogs/Валюты",
                 "create_from_form": True,
                 "evidence": "dep_zakaz_clienta.json"},
                {"node_id": "ERPcode/Catalogs/Склады (Кладовая собственного магазина — для собственного магазина)",
                 "create_from_form": False,
                 "evidence": "ANS.all_sklsady_confirmed + q05_own_store"},
                {"node_id": "ERPcode/Catalogs/Номенклатура",
                 "create_from_form": True,
                 "evidence": "ANS.polufabrikat_separate_nomenclature + сырьё + ГП"},
                {"node_id": "ERPcode/Catalogs/Упаковки",
                 "create_from_form": True,
                 "evidence": "ANS.hleb_units_upakovki"},
                {"node_id": "ERPcode/Catalogs/Приоритеты",
                 "create_from_form": False,
                 "evidence": "dep_zakaz_clienta.json"},
                {"node_id": "ERPcode/Catalogs/ВидыЦен",
                 "create_from_form": False,
                 "evidence": "dep_zakaz_clienta.json (через Соглашение→ВидЦен)"},
            ],
            "key_attributes": [
                "Партнер (Клиент) — создается в форме",
                "Контрагент — создается в форме",
                "Соглашение об условиях продаж — создается в форме (для каждого типа клиента свое, подтверждено audit_q02)",
                "Договор — создается в форме",
                "Организация",
                "Валюта — создается в форме",
                "Склад отгрузки (Кладовая собственного магазина / Склад ГП)",
                "Приоритет",
                "Хозяйственная операция = Реализация / Реализация через комиссию",
                "НалогообложениеНДС",
                "Способ доставки (Самовывоз / Доставка — подтверждено ANS.samovyvoz_flag)",
                "Статус = 'К производству' (подтверждено ANS.dannye_zakazklient)",
                "ДатаОтгрузки = Дата производства + 1 день (подтверждено constraint 14)",
                "ГрафикДоставки / СрокОтгрузки (если доставка, ANS.grafik_dostavki)",
                "ТЧ 'Товары': Номенклатура (вид «Готовая продукция»), Характеристика, Упаковка, Количество (шт.), Цена (по ВидуЦен)",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ЗаказПоставщику", "relation": "не предусмотрен"},
                {"created_doc": "РеализацияТоваровУслуг", "relation": "Создать на основании → Реализация",
                 "evidence": "chain_klient_realiz.json (ЗаказКлиента → Реализация)"}
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыКОтгрузке",
                 "registrator": "ЗаказКлиента (статус согласован)"},
                {"register": "ERPcode/AccumulationRegisters/РасчетыСКлиентамиПланОтгрузок",
                 "registrator": "ЗаказКлиента"},
                {"register": "ERPcode/AccumulationRegisters/РасчетыСКлиентамиПланОплат",
                 "registrator": "ЗаказКлиента"},
                {"register": "ERPcode/AccumulationRegisters/ЗапасыИПотребности",
                 "registrator": "ЗаказКлиента"},
                {"register": "ERPcode/AccumulationRegisters/РаспоряженияНаОтгрузкуИВозврат",
                 "registrator": "ЗаказКлиента"},
            ],
            "controls": [
                "Проверка Дата подачи < 12:00 — контроль не автоматизирован в метаданных; регламентная проверка на стороне бизнес-логики (constraint:14, control:12, подтверждение ANS.manager_confirmation).",
                "Проверка Номенклатура-Упаковка-Количество (Самовывоз/Доставка) — проверяется при заполнении ТЧ «Товары», подтверждено ANS.edit_nakladnaya",
                "Принять/частично/изолировать/вернуть — неприменимо; применяется в приемке сырья (block 4)",
            ],
            "alt_branches": [
                {"condition": "После 12:00 — срочная заявка",
                 "graph_gap": "question:gap:12b2d529a863893f — заказчик подтвердил: срочные после 12:00 не принимаются (constraint:15)"},
                {"condition": "Клиент не существует",
                 "handling": "Создание Партнера+Контрагента+Соглашения из формы заказа",
                 "evidence": "dep_zakaz_clienta.json inline-candidate=true"}
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ЗаказКлиента (статус 'К производству')",
                "key_for_next": ["Номенклатура", "Количество (шт.)", "ДатаОтгрузки", "Склад"]
            },
        }
    ],
    "outputs": [
        "Зарегистрированная и подтвержденная заявка клиента (ЗаказКлиента, статус «К производству»)",
        "Переданные данные для формирования производственного плана (через регистр ЗапасыИПотребности, см. gap-планирование)",
    ]
}

block2 = {
    "block_id": 2,
    "title": "ПЛАНИРОВАНИЕ ПРОИЗВОДСТВА",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:2",
    "scenario": {"id": "scenario_010--2.10. Планирование производства",
                 "title": "2.10. Планирование производства",
                 "layer": 1},
    "summary": (
        "Согласованные ЗаказыКлиента конвертируются в ЗаказНаПроизводство2_2 по каждому клиенту "
        "(на 1–2 дня, подтверждено ANS.zakaz_na_proizv_each). ПланПроизводства по "
        "прежнему ведётся как сводный план-график, но реальным контуром исполнения служит "
        "ЗаказНаПроизводство2_2+ЭтапПроизводства2_2."
    ),
    "open_gaps": ["question:gap:3336ef51947e7bfc — автозаполнение ЗаказовНаПроизводство",
                  "block:2 — выбор по [scenario_003] 5.3 неоднозначен (mapping_gaps)"],
    "steps": [
        {
            "step_id": "2.1",
            "name": "Формирование производственного плана на послезавтра",
            "event": "Подтвержденные заявки клиентов агрегируются в план",
            "primary_document": {
                "node_id": "ERPcode/Documents/ПланПроизводства",
                "title": "Документ План производства (сводный)",
                "ui_paths": [
                    "БюджетированиеИПланирование → ПланированиеЗапасов → ПланПроизводства",
                    "Производство → МежцеховоеУправление2_1 → ПланПроизводства",
                    "Производство → МежцеховоеУправление2_1_СЗаголовком → ПланПроизводства",
                ],
                "evidence": "process_plan.json operational_steps[1].ui_paths + dep_plan_proizv.json",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/ВидыПланов",
                 "create_from_form": False,
                 "evidence": "dep_plan_proizv.json (ВидПлана)"},
                {"node_id": "ERPcode/Catalogs/СценарииТоварногоПланирования",
                 "create_from_form": False,
                 "evidence": "dep_plan_proizv.json (Сценарий)"},
                {"node_id": "ERPcode/Catalogs/Структура предприятия",
                 "create_from_form": False,
                 "evidence": "dep_plan_proizv.json (Подразделение)"},
                {"node_id": "ERPcode/Catalogs/Номенклатура",
                 "create_from_form": False,
                 "evidence": "ТЧ «Продукция»"},
            ],
            "key_attributes": [
                "ВидПлана (тип производственного процесса)",
                "Сценарий (план-факт)",
                "Подразделение (в нашем случае — два цеха, см. block 6/7)",
                "Период (день/неделя)",
                "ТЧ «Продукция» — Номенклатура из ЗаказаКлиента, Количество (шт.)",
                "Статус = 'Утвержден'",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ЗаказНаПроизводство2_2",
                 "relation": "Создать на основании (вручную или автозаполнение, ANS.autozapoln_zakazov)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ПланыВыпускаИзделий",
                 "registrator": "ПланПроизводства (при проведении)",
                 "evidence": "dep_plan_proizv.json"},
                {"register": "ERPcode/AccumulationRegisters/ОборотыБюджетов / ФактическиеДанныеБюджетирования",
                 "registrator": "ПланПроизводства"},
            ],
            "controls": [
                "Соответствие плана поступившим заявкам — обеспечивается ЗаказКлиента→ЗаказНаПроизводство2_2 (audit GAP-6)",
                "Достаточность сырья — отчёт «Потребности в материалах» / «Запасы и потребности» (ANS.samovyvoz)",
            ],
            "alt_branches": [
                {"condition": "Сырья не хватает",
                 "handling": "Блок 3 — формирование заказов поставщику по потребностям "
                             "(scenario_018--2.18. Формирование заказов поставщикам по планам)"},
                {"condition": "Make-to-Order",
                 "handling": "Объем производства = объем заявок, без запаса "
                             "(constraint:32,33 + ANS.mto_only)"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ЗаказНаПроизводство2_2 (по каждому ЗаказуКлиента)",
                "key_for_next": ["Подразделения (холодный/горячий цех) → см. block 6/7"]
            },
        },
        {
            "step_id": "2.2",
            "name": "Автозаполнение ЗаказовНаПроизводство по ЗаказамКлиента",
            "event": "Создание оперативного заказа на производство 1–2 дня",
            "primary_document": {
                "node_id": "ERPcode/Documents/ЗаказНаПроизводство2_2",
                "title": "Документ Заказ на производство (2.2)",
                "ui_paths": [
                    "Производство → МежцеховоеУправление2_2 → ЗаказНаПроизводство2_2",
                ],
                "evidence": "ANS.zakaz_na_proizv_each + gap:e14ac56889d24b95",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Структура предприятия (холодный/горячий цех)",
                 "create_from_form": False,
                 "evidence": "ANS.etap_22_two_podrazdeleniya (block 6/7)"},
                {"node_id": "ERPcode/Catalogs/Номенклатура (ГП, Характеристики)",
                 "create_from_form": False,
                 "evidence": "ANS.polufabrikat_separate_nomenclature"},
                {"node_id": "ERPcode/Catalogs/РесурсныеСпецификации",
                 "create_from_form": True,
                 "evidence": "ANS.res_spec + ANS.res_spec_yes + ANS.res_spec_yes2 (полуфабрикат Тесто)"},
                {"node_id": "ERPcode/Catalogs/ЭтапыПроизводства",
                 "create_from_form": True,
                 "evidence": "ANS.etap22_two_podr; Холодный → Горячий"},
            ],
            "key_attributes": [
                "Статус = 'К производству'",
                "Подразделение-исполнитель (опционально, задаётся на этапах)",
                "ТЧ «Этапы»: Подразделение, ДатаНачала, ДатаОкончания, РесурснаяСпецификация",
                "ТЧ «Продукция»: Номенклатура (ГП), Количество (шт.), РесурснаяСпецификация",
                "ТЧ «Полуфабрикаты»: номенклатура 'Тесто', Количество (по рецептуре)",
                "ТЧ «Материалы»: сырьё (мука, дрожжи, соль, сахар, семена)",
                "Назначение = ЗаказКлиента (основание)",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ЭтапПроизводства2_2",
                 "relation": "Создать на основании (по каждому подразделению)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ПланыВыпускаИзделий",
                 "registrator": "ЗаказНаПроизводство2_2",
                 "evidence": "dep_zakaz_proizv_22.json + sources"},
                {"register": "ERPcode/AccumulationRegisters/ТоварыКОтгрузке (через связь с ЗаказКлиента)",
                 "registrator": "ЗаказНаПроизводство2_2"},
            ],
            "controls": [
                "Автозаполнение ЗаказовНаПроизводство (ANS.autozapoln_zakazov) — через обработку Формирование заказов по потребностям",
                "Контроль через 5.3 Формирование заказов по потребностям (ANS.form_zakazov_5_3)",
                "Контроль в 1С:ERP (ANS.kontrol_1c_erp), без Excel/внешних систем",
            ],
            "alt_branches": [
                {"condition": "Не заполнена ресурсная спецификация",
                 "handling": "Блокирующая ошибка — этап не запустится (см. block 6)"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ЭтапПроизводства2_2 (холодный цех, затем горячий цех)",
                "key_for_next": ["Подразделение (цех)", "ДатаНачала", "ДатаОкончания", "РесурснаяСпецификация"]
            },
        },
    ],
    "outputs": [
        "Производственный план на послезавтра (ПланПроизводства + связка ЗаказовНаПроизводство2_2)",
        "Распределение заказов по цехам (ЭтапПроизводства2_2 по подразделениям)",
    ]
}

block3 = {
    "block_id": 3,
    "title": "ПЛАНИРОВАНИЕ ЗАКУПОК СЫРЬЯ",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:3",
    "scenario": {"id": "scenario_012--2.12. Планирование закупок",
                 "title": "2.12. Планирование закупок",
                 "layer": 1,
                 "confidence": 0.8},
    "summary": (
        "Закупка не под каждый заказ клиента, а по среднему расходу за N дней с "
        "учётом минимальных лимитов (позиция+склад+период, ANS.min_zapas_pos_sk_period). "
        "При падении ниже лимита формируется ЗаказПоставщику, направляемый одному из "
        "двух поставщиков (быстрый/мелкие позиции; долгий/крупные партии)."
    ),
    "open_gaps": ["question:gap:1e3b9440a582211e (проверка наличия сырья — авт.)",
                  "question:gap:984d48cedec2021d — 5.3 уже разрешено (a)"],
    "steps": [
        {
            "step_id": "3.1",
            "name": "Контроль остатков сырья и лимитов",
            "event": "Отчет по среднему расходу за N дней и текущим остаткам",
            "primary_document": {
                "node_id": "ERPcode/Reports/ОстаткиТоваровНаСкладах",
                "title": "Отчет Остатки товаров на складах",
                "evidence": "ANS.tovary_na_skl — отчёт для контроля",
            },
            "required_nsi": [
                {"node_id": "ERPcode/InformationRegisters/ЦеныНоменклатурыПоставщиков",
                 "create_from_form": False,
                 "evidence": "dep_zakaz_post.json (для автозаполнения цены)"},
                {"node_id": "ERPcode/Catalogs/Склады (склад сырья + кладовые цехов)",
                 "create_from_form": False,
                 "evidence": "ANS.all_sklsady_confirmed"},
            ],
            "key_attributes": [
                "Отчётный период (N дней, ANS.rashod_avg_N_days)",
                "Позиция номенклатуры (сырьё: мука, дрожжи, соль, сахар, семена)",
                "Текущий остаток на складе сырья",
                "Минимальный запас (настройка: позиция+склад+период)",
                "Среднесуточный расход = среднее за N дней",
                "Точка заказа = МинЗапас + Расход × (ДлитПоставки + СтраховойЗапас)",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ЗаказПоставщику",
                 "relation": "Создать на основании (через сценарий 5.3 Формирование заказов по потребностям, ANS.form_zakazov_5_3)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "регистратор ПриобретениеТоваровУслуг (заполняется в block 4)",
                 "evidence": "ANS.tovary_na_skl"},
            ],
            "controls": [
                "Контроль лимитов (control:51) — отчёт «Остатки товаров на складах» с фильтром «ниже минимума»",
                "Своевременность заказа (control:52) — еженедельно: задание планировщика",
                "Достаточность для плана производства (control:30) — через отчёт «Себестоимость товаров» / «Потребности в материалах»",
            ],
            "alt_branches": [
                {"condition": "Запас ниже минимума",
                 "handling": "Заказ поставщику с приоритетом 1 → быстрый поставщик "
                             "(мелкие позиции, поставка 1 раз/нед)"},
                {"condition": "Крупная партия",
                 "handling": "Заказ поставщику с приоритетом 2 → долгий поставщик "
                             "(крупные партии, поставка несколько раз/мес)"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ЗаказПоставщику (статус 'Согласован')",
                "key_for_next": ["Партнер (Поставщик 1/2)", "Номенклатура+Количество", "Склад (Склад сырья)", "ДатаПоставки"]
            },
        },
        {
            "step_id": "3.2",
            "name": "Формирование и отправка ЗаказаПоставщику",
            "event": "Автогенерация/ручное создание заказа по потребности",
            "primary_document": {
                "node_id": "ERPcode/Documents/ЗаказПоставщику",
                "title": "Документ Заказ поставщику",
                "ui_paths": [
                    "Закупки → Закупки → ЗаказПоставщику",
                    "Закупки → РасчетыСПоставщиками → ЗаказПоставщику",
                    "ЗакупкиБазовая → ВедениеЗаказовБазовая → ЗаказПоставщику"
                ],
                "evidence": "process_plan.json operational_steps[2].ui_paths",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Партнеры (Поставщик 1 / 2)",
                 "create_from_form": True,
                 "evidence": "dep_zakaz_post.json (inline-candidate)"},
                {"node_id": "ERPcode/Catalogs/Контрагенты",
                 "create_from_form": True, "evidence": "dep_zakaz_post.json"},
                {"node_id": "ERPcode/Catalogs/ДоговорыКонтрагентов",
                 "create_from_form": True, "evidence": "dep_zakaz_post.json"},
                {"node_id": "ERPcode/Catalogs/Валюты", "create_from_form": True, "evidence": "dep_zakaz_post.json"},
                {"node_id": "ERPcode/Catalogs/Соглашения об условиях закупок",
                 "create_from_form": True, "evidence": "dep_zakaz_post.json (через схожее поле)"},
                {"node_id": "ERPcode/Catalogs/Номенклатура (сырьё)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/Catalogs/Склады (Склад сырья)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
            ],
            "key_attributes": [
                "Партнер (Поставщик 1 или 2)",
                "Контрагент",
                "Соглашение об условиях закупок",
                "Договор (вид: С поставщиком, ТипДоговора=Закупка)",
                "Валюта",
                "ХозяйственнаяОперация = Закупка у поставщика",
                "Склад (Склад сырья)",
                "Организация",
                "ТЧ «Товары»: Номенклатура (мука, дрожжи, соль, сахар, семена), Количество, Цена (по ценам поставщика, dep_zakaz_post.json)",
                "Статус = 'Согласован' → 'К поступлению'",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ПриобретениеТоваровУслуг",
                 "relation": "Создать на основании → ПриобретениеТоваровУслуг",
                 "evidence": "chain_post_priobr.json (ЗаказПоставщику → Приобретение)"},
                {"created_doc": "АктКонтроляКачестваТоваров",
                 "relation": "при поставке → блок 4"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/РасчетыСПоставщикамиПланПоставок",
                 "registrator": "ЗаказПоставщику",
                 "evidence": "dep_zakaz_post.json + chain_post_priobr.json"},
                {"register": "ERPcode/AccumulationRegisters/РасчетыСПоставщикамиПланОплат",
                 "registrator": "ЗаказПоставщику"},
                {"register": "ERPcode/InformationRegisters/ЦеныНоменклатурыПоставщиков",
                 "registrator": "ЗаказПоставщику (заполнение при согласовании)"},
                {"register": "ERPcode/AccumulationRegisters/ЗапасыИПотребности",
                 "registrator": "ЗаказПоставщику"},
            ],
            "controls": [
                "Контроль минимальных лимитов (control:51, ANS.min_zapas_pos_sk_period)",
                "Своевременность заказа (control:52)",
                "Соответствие плану производства (control:30)",
                "Способ размещения (быстрый/долгий поставщик) — настройка в Соглашении",
            ],
            "alt_branches": [
                {"condition": "У поставщика нет в наличии",
                 "handling": "Перенос заказа альтернативному поставщику (если Соглашение «допускает замену»)"},
                {"condition": "Заказ отменен",
                 "handling": "Закрытие ЗаказПоставщику (статус 'Отменен'), перепланирование потребности"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ЗаказПоставщику (статус 'К поступлению')",
                "key_for_next": ["ДатаПоставки", "Номенклатура+Количество", "Склад"]
            },
        },
    ],
    "outputs": [
        "Сформированный и отправленный заказ поставщику (ЗаказПоставщику)",
        "Плановые потребности в сырье (через регистр ЗапасыИПотребности)",
    ]
}

block4 = {
    "block_id": 4,
    "title": "ВХОДНОЙ КОНТРОЛЬ И ПРИЕМКА СЫРЬЯ",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:4",
    "scenario": {"id": "scenario_003--6.3. Приемка товаров на склад",
                 "title": "6.3. Приемка товаров на склад",
                 "layer": 1,
                 "confidence": "audit-corrected"},
    "summary": (
        "Поставка по ЗаказуПоставщику поступает на склад сырья; кладовщик проводит "
        "АктКонтроляКачестваТоваров (5 состояний: принять/частично/изолировать/"
        "вернуть/списать). Годное сырьё приходуется ПриобретениемТоваровУслуг с "
        "указанием серий и срока годности, особенно для муки."
    ),
    "open_gaps": ["question:gap:db8f5f0f20e9d70c — АктКонтроляКачестваТоваров (подтверждено a)",
                  "question:gap:7322f5f216d16764 — автоматическая проверка через настройку (подтверждено a)",
                  "gap:AE49CD — 5 состояний приемки (подтверждено)"],
    "steps": [
        {
            "step_id": "4.1",
            "name": "Входной контроль качества и оформление акта",
            "event": "Кладовщик осматривает сырьё и фиксирует результат в Акте",
            "primary_document": {
                "node_id": "ERPcode/Documents/АктКонтроляКачестваТоваров",
                "title": "Документ Акт контроля качества товаров",
                "evidence": "ANS.akt_kontrolya_kach (a)",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Номенклатура (сырьё)",
                 "create_from_form": False, "evidence": "ЗаказПоставщику"},
                {"node_id": "ERPcode/Catalogs/Склады (Склад сырья)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры (для муки)",
                 "create_from_form": True, "evidence": "ANS.serii_systemic"},
            ],
            "key_attributes": [
                "Основание = ЗаказПоставщику",
                "ТЧ «Товары»: Номенклатура, Количество, СостояниеПриемки",
                "СостояниеПриемки = Принять / ПринятьЧастично / Изолировать / Вернуть / Списать",
                "ЛабораторныйАнализ = Да/Нет (для муки — обязательно)",
                "Дата начала/окончания анализа",
                "Ответственный = Кладовщик (ANS.quality_control_owner)",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ПриобретениеТоваровУслуг",
                 "relation": "Создать на основании (только для СостояниеПриемки = Принять/Частично)",
                 "evidence": "chain_akk_priobr.json"},
                {"created_doc": "ВозвратТоваровПоставщику",
                 "relation": "Создать на основании (для СостояниеПриемки = Вернуть)",
                 "evidence": "ANS.ae49cd3744ac0838 (частичная приемка через возврат)"},
                {"created_doc": "СписаниеТоваров / Изолирование склада",
                 "relation": "Создать на основании (для Списать / Изолировать) — graph_gap: ERPcode/Documents/АктОСписанииТоваровУТМ (см. блок gaps)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ПриобретениеТоваровУслуг (после принятия)",
                 "evidence": "ANS.tovary_na_skl"},
            ],
            "controls": [
                "АктКонтроляКачестваТоваров обязателен при приемке (control:69, ANS.akt_kontrolya_kach)",
                "Автоматическая проверка через настройку «ИспользоватьКонтрольКачества» = Истина (ANS.auto_proverka_nastroyka)",
                "Время срабатывания — кладовщик при приеме (ANS.quality_control_owner)",
                "Только лаборатория для муки — visual+docs (control:69 + original_text block:4:action:62)",
            ],
            "alt_branches": [
                {"condition": "Мука не соответствует",
                 "handling": "ВозвратТоваровПоставщику + служебная записка (5-состояние)"},
                {"condition": "Сырье частично годно",
                 "handling": "ПринятьЧастично → ВозвратТоваровПоставщику на разницу"},
                {"condition": "Сырье изолировано",
                 "handling": "Карантинная ячейка (graph_gap на соответствующий документ)"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ПриобретениеТоваровУслуг (только годное)",
                "key_for_next": ["Номенклатура+Количество принятое", "Серии (срок годности)"]
            },
        },
        {
            "step_id": "4.2",
            "name": "Оприходование годного сырья на склад сырья",
            "event": "Принятое сырьё приходуется с записью в ТоварыНаСкладах",
            "primary_document": {
                "node_id": "ERPcode/Documents/ПриобретениеТоваровУслуг",
                "title": "Документ Приобретение товаров и услуг",
                "ui_paths": [
                    "Закупки → Закупки → ПриобретениеТоваровУслуг",
                ],
                "evidence": "ANS.priobr_tov_uslug (да)",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Склады (Склад сырья)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры",
                 "create_from_form": True, "evidence": "ANS.serii_systemic (b)"},
                {"node_id": "ERPcode/Catalogs/ВидыЦен (по средней закупочной, ANS.2e06a323d8173c77)",
                 "create_from_form": False, "evidence": "ANS.price_average_purchase"},
            ],
            "key_attributes": [
                "Основание = ЗаказПоставщику + АктКонтроляКачестваТоваров",
                "Партнер/Контрагент/Соглашение/Договор",
                "Склад = Склад сырья",
                "ТЧ «Товары»: Номенклатура, Серия (срок годности для муки), Количество, Цена, Сумма",
                "Цена по средней закупочной (ANS.price_average_purchase)",
                "Подразделение/Организация",
                "Дата = ДатаПоставки",
                "Статус = 'Принят', флаг 'Проведен в ERP'",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ПеремещениеТоваров",
                 "relation": "на кладовую холодного/горячего цеха по потребности блока 6/7"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ПриобретениеТоваровУслуг (приход)",
                 "evidence": "ANS.tovary_na_skl"},
                {"register": "ERPcode/AccumulationRegisters/ПартииТоваровОрганизаций",
                 "registrator": "ПриобретениеТоваровУслуг",
                 "evidence": "ANS.uchet_po_partiyam (да)"},
                {"register": "ERPcode/AccumulationRegisters/РасчетыСПоставщиками",
                 "registrator": "ПриобретениеТоваровУслуг",
                 "evidence": "dep_priobr.json"},
            ],
            "controls": [
                "Оформление документов на оприходование (control:70)",
                "Контроль качества (control:69, пройден на 4.1)",
                "Ордерная схема приёмки (ANS.ordernaya_schema)",
                "Учёт по партиям (ANS.uchet_po_partiyam)",
            ],
            "alt_branches": [
                {"condition": "Расхождение факт/ЗаказПоставщику",
                 "handling": "АктОРасхождениях после ПриобретенияТоваровУслуг с корректировкой (graph_gap: документ АктОРасхождениях)"},
                {"condition": "Партия просрочена",
                 "handling": "СписаниеТоваров / ВозвратТоваровПоставщику, без приходования"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Приход по ПриобретениюТоваровУслуг в ТоварыНаСкладах (Склад сырья)",
                "key_for_next": ["Остатки по позиции сырья + Серии + Срок годности"]
            },
        },
    ],
    "outputs": [
        "Оприходованное сырье на складе (годное к использованию)",
        "Акт возврата (если сырье забраковано) / акт списания (если списано)",
        "Скорректированные регистры: ТоварыНаСкладах, ПартииТоваровОрганизаций, РасчетыСПоставщиками",
    ]
}

block5 = {
    "block_id": 5,
    "title": "СКЛАДСКОЕ ХОЗЯЙСТВО (ХРАНЕНИЕ СЫРЬЯ И МАТЕРИАЛОВ)",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:5",
    "scenario": {"id": None,
                 "title": None,
                 "layer": None,
                 "confidence": "graph_gap"},
    "summary": (
        "Сырьё хранится на Складе сырья (мука, дрожжи, соль, сахар, семена). "
        "Упаковочные материалы и многооборотная тара — на Складе хозинвентаря и "
        "упаковки (последняя не используется ANS.c7ed83841136dbe2). Кладовые при цехах "
        "обеспечивают текущие потребности производства. ФИФО через серии."
    ),
    "open_gaps": ["graph_gap: 5 storage logs (не автоматизировано в 1С, регламентируется ордерной схемой)",
                  "ANS.c7ed83841136dbe2 — многооборотная тара не используется → этап упрощается"],
    "steps": [
        {
            "step_id": "5.1",
            "name": "Размещение сырья на складе сырья",
            "event": "Оприходованное сырьё размещается в зоне хранения",
            "primary_document": {
                "node_id": "ERPcode/Documents/ОтборРазмещениеТоваров",
                "title": "Документ Отбор (размещение) товаров",
                "evidence": "ANS.ordernaya_schema (ордерная схема везде) + ANS.sklsady = Склад сырья",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Склады (Склад сырья)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/Catalogs/СкладскиеЯчейки (опционально — при ячеечном хранении)",
                 "create_from_form": False, "evidence": "graph_gap (см. блок gaps)"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры (срок годности)",
                 "create_from_form": True, "evidence": "ANS.serii_systemic"},
            ],
            "key_attributes": [
                "Склад (Склад сырья)",
                "ТЧ «Товары»: Номенклатура, Серия, Количество, Ячейка",
                "Зона хранения (Сухая / Охлаждаемая — для дрожжей)",
                "Дата поступления, Дата истечения срока годности",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ПеремещениеТоваров",
                 "relation": "Создать на основании для выдачи в кладовую цеха",
                 "evidence": "ANS.perem_tov_yes (да)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ОтборРазмещениеТоваров (размещение)",
                 "evidence": "ANS.sklsady + ANS.tovary_na_skl"},
            ],
            "controls": [
                "Условия хранения (control:86)",
                "Контроль остатков (control:87) — отчёт ОстаткиТоваровНаСкладах",
                "Метод ФИФО (constraint:89, ANS.fifo_policy)",
                "Срок годности (constraint:90, ANS.serii_systemic)",
            ],
            "alt_branches": [
                {"condition": "Истекает срок годности муки",
                 "handling": "Перемещение в зону 'на списание' + СписаниеТоваров / ВозвратТоваровПоставщику (graph_gap: документ списания)"},
                {"condition": "Тара не используется",
                 "handling": "Этап пропускается (ANS.c7ed83841136dbe2)"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Запас по сериям с ФИФО-критерием",
                "key_for_next": ["Доступные серии (самый ранний срок для ФИФО)"]
            },
        },
        {
            "step_id": "5.2",
            "name": "Выдача сырья в кладовые цехов (холодного/горячего)",
            "event": "По потребности производственного плана сырьё перемещается в кладовые цехов",
            "primary_document": {
                "node_id": "ERPcode/Documents/ПеремещениеТоваров",
                "title": "Документ Перемещение товаров",
                "evidence": "ANS.perem_tov_yes (да) + ANS.etap22_perem (ЭтапПроизводства2_2 + ПеремещениеТоваров)",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Склады (Склад-отправитель=Склад сырья, Склад-получатель=Кладовая холодного цеха / Кладовая горячего цеха)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры",
                 "create_from_form": False, "evidence": "перенос серии"},
            ],
            "key_attributes": [
                "Склад-отправитель / Склад-получатель",
                "Дата перемещения = ДатаЗапускаЭтапа - 0.5 дня (для подготовки)",
                "ТЧ «Товары»: Номенклатура, Серия, Количество (кг, л)",
                "Назначение = ЭтапПроизводства2_2 (опционально)",
                "Ответственный = Кладовщик цеха",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ЭтапПроизводства2_2",
                 "relation": "Создать на основании → ЭтапПроизводства2_2 (по данным о потребности)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ПеремещениеТоваров (расход со склада сырья / приход на кладовую)"},
                {"register": "ERPcode/AccumulationRegisters/ДвиженияСерийТоваров",
                 "registrator": "ПеремещениеТоваров"},
            ],
            "controls": [
                "Условия хранения (control:86)",
                "Контроль остатков (control:87)",
                "ФИФО (constraint:89, ANS.fifo_policy)",
                "Время в кладовой: за 0.5–1 день до старта этапа",
            ],
            "alt_branches": [
                {"condition": "Сырья в кладовой не хватает",
                 "handling": "Возврат на block 3 — заказ поставщику срочно"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Сырьё на кладовой холодного цеха",
                "key_for_next": ["Доступный запас с ФИФО-серией"]
            },
        },
    ],
    "outputs": [
        "Сырье и материалы размещены по складским зонам",
        "Обеспечена сохранность и учет запасов (ФИФО)",
        "Сырьё готово к запуску в этап производства",
    ]
}

block6 = {
    "block_id": 6,
    "title": "ПРОИЗВОДСТВО ПОЛУФАБРИКАТА (ХОЛОДНЫЙ ЦЕХ)",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:6",
    "scenario": {"id": "scenario_010--9.2. Производство 2.2",
                 "title": "9.2. Производство 2.2 (межцеховое)",
                 "layer": 1,
                 "confidence": "confirmed by user etap22_two_podrazdeleniya"},
    "summary": (
        "ЭтапПроизводства2_2 в холодном цехе (Подразделение = Холодный цех). По "
        "ресурсной спецификации идёт замес теста (5 материалов по 5 видам хлеба). "
        "Выпуск полуфабриката 'Тесто' на Кладовую полуфабрикатов / горячий цех."
    ),
    "open_gaps": ["graph_gap: спецификации по 5 видам хлеба (каждая имеет свою ресурсную спецификацию + полуфабрикат Тесто по виду хлеба)",
                  "graph_gap: документ 'Передача полуфабриката в горячий цех' = ЭтапПроизводства2_2 (b)"],
    "steps": [
        {
            "step_id": "6.1",
            "name": "Запуск ЭтапаПроизводства в холодном цехе",
            "event": "По ЗаказуНаПроизводство2_2 создаётся ЭтапПроизводства2_2 в подразделении «Холодный цех»",
            "primary_document": {
                "node_id": "ERPcode/Documents/ЭтапПроизводства2_2",
                "title": "Документ Этап производства 2.2",
                "ui_paths": [
                    "Производство → МежцеховоеУправление2_2 → ЭтапыПроизводства2_2",
                ],
                "evidence": "ANS.etap22_two_podrazdeleniya + ANS.gap:60685ff216f76dba",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Структура предприятия (Холодный цех)",
                 "create_from_form": False, "evidence": "ANS.etap22_two_podrazdeleniya"},
                {"node_id": "ERPcode/Catalogs/РесурсныеСпецификации (5 спецификаций по 5 видам хлеба)",
                 "create_from_form": True,
                 "evidence": "ANS.res_spec + ANS.res_spec_yes + ANS.res_spec_yes2 (полуфабрикат Тесто)"},
                {"node_id": "ERPcode/Catalogs/ЭтапыПроизводства (этап 'Замес теста')",
                 "create_from_form": True, "evidence": "ANS.etap22_two_podr"},
                {"node_id": "ERPcode/Catalogs/Склады (Кладовая холодного цеха — приход сырья; Кладовая полуфабрикатов — выпуск)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/Catalogs/Номенклатура (Тесто — отдельная номенклатура, вид «Полуфабрикат»)",
                 "create_from_form": True, "evidence": "ANS.polufabrikat_separate_nomenclature (a)"},
            ],
            "key_attributes": [
                "ЗаказНаПроизводство2_2 (основание)",
                "Подразделение = 'Холодный цех'",
                "РесурснаяСпецификация (для текущего вида хлеба: батон/вс/бородинский/слойка/лепешка)",
                "ЭтапПроизводства (Замес теста)",
                "Склад получения сырья = Кладовая холодного цеха",
                "Склад выпуска = Кладовая полуфабрикатов / Горячий цех (для следующего этапа)",
                "ТЧ «ВыходныеИзделия»: Номенклатура (Тесто), Количество (кг)",
                "ТЧ «Материалы»: сырьё (мука, дрожжи, соль, сахар, семена), Серия, Количество",
                "ДатаНачала / ДатаОкончания",
                "Статус → 'Завершён'",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ПеремещениеТоваров",
                 "relation": "Создать на основании для передачи Теста в горячий цех",
                 "evidence": "ANS.etap22_perem (да)"},
                {"created_doc": "ЭтапПроизводства2_2 (следующий, в горячем цехе)",
                 "relation": "по цепочке в block 7"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ЭтапПроизводства2_2 (расход сырья, приход полуфабриката)",
                 "evidence": "dep_etap_22.json"},
                {"register": "ERPcode/AccumulationRegisters/ПартииПроизводства",
                 "registrator": "ЭтапПроизводства2_2",
                 "evidence": "audit GAP-8 (tag через регистр партий)"},
            ],
            "controls": [
                "Контроль качества теста (control:106, ANS.quality_control_owner — можно делегировать мастеру цеха)",
                "Точное соответствие ресурсной спецификации (контроль через отчёт «Плановая себестоимость продукции»)",
                "ФИФО по сериям сырья (constraint:89, ANS.fifo_policy)",
            ],
            "alt_branches": [
                {"condition": "Тесто забраковано",
                 "handling": "Списание брака + возврат бракованного сырья в производство (graph_gap на СписаниеТоваров + ДвижениеПродукцииИМатериалов)"},
                {"condition": "Не хватает сырья",
                 "handling": "Прерывание этапа + срочный заказ поставщику → block 3"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ЭтапПроизводства2_2 (Тесто — партия производства)",
                "key_for_next": ["Номенклатура Тесто + Серия партии", "Количество (кг)"]
            },
        },
        {
            "step_id": "6.2",
            "name": "Передача полуфабриката Тесто в горячий цех",
            "event": "Перемещение теста на Кладовую горячего цеха",
            "primary_document": {
                "node_id": "ERPcode/Documents/ПеремещениеТоваров",
                "title": "Документ Перемещение товаров",
                "evidence": "ANS.perem_tov_yes + ANS.etap22_perem",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Склады (Кладовая полуфабрикатов → Кладовая горячего цеха)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
            ],
            "key_attributes": [
                "Склад-отправитель = Кладовая полуфабрикатов / Кладовая холодного цеха",
                "Склад-получатель = Кладовая горячего цеха",
                "ТЧ «Товары»: Номенклатура (Тесто), Серия партии, Количество (кг)",
                "Дата перемещения = ЗавершениеЭтапа",
                "Назначение = ЭтапПроизводства2_2 (горячий цех)",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ЭтапПроизводства2_2 (горячий цех)",
                 "relation": "Создать на основании (block 7.1)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ПеремещениеТоваров"},
            ],
            "controls": [
                "Контроль сроков между этапами (Тесто — короткий срок жизни)",
                "Контроль количества (Тесто при переходе) = выход РесурснойСпецификации",
            ],
            "alt_branches": [
                {"condition": "Задержка между цехами",
                 "handling": "Документирование в ОтборРазмещениеТоваров + оповещение мастера горячего цеха"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Тесто на Кладовой горячего цеха",
                "key_for_next": ["Готовность к выпечке по каждому виду хлеба"]
            },
        },
    ],
    "outputs": [
        "Готовый полуфабрикат (тесто) для каждого вида продукции",
        "Тесто передается в горячий цех (кладовую горячего цеха)",
    ]
}

block7 = {
    "block_id": 7,
    "title": "ПРОИЗВОДСТВО ГОТОВОЙ ПРОДУКЦИИ (ГОРЯЧИЙ ЦЕХ)",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:7",
    "scenario": {"id": "scenario_010--9.2. Производство 2.2",
                 "title": "9.2. Производство 2.2 (межцеховое) — горячий цех",
                 "layer": 1,
                 "confidence": "confirmed by user etap22_two_podrazdeleniya"},
    "summary": (
        "ЭтапПроизводства2_2 в горячем цехе (Подразделение = Горячий цех). По "
        "ресурсной спецификации идёт формовка, выпечка, охлаждение ГП (5 видов: батон "
        "белый, хлеб белый в/с, бородинский, слойка с изюмом, лепешка). Выпуск 5 видов "
        "ГП на Склад готовой продукции. Серия хлеба = дата производства + срок 5 дней."
    ),
    "open_gaps": ["graph_gap: документ ВыпускПродукции vs ЭтапПроизводства2_2 (chain не найден)"],
    "steps": [
        {
            "step_id": "7.1",
            "name": "Запуск ЭтапаПроизводства в горячем цехе (формование, выпечка, охлаждение)",
            "event": "Тесто передаётся в горячий цех → создаётся ЭтапПроизводства2_2",
            "primary_document": {
                "node_id": "ERPcode/Documents/ЭтапПроизводства2_2",
                "title": "Документ Этап производства 2.2 (горячий цех)",
                "ui_paths": [
                    "Производство → МежцеховоеУправление2_2 → ЭтапыПроизводства2_2",
                ],
                "evidence": "ANS.gap:60685ff216f76dba + ANS.finish_in_gorach",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Структура предприятия (Горячий цех)",
                 "create_from_form": False, "evidence": "ANS.etap22_two_podrazdeleniya"},
                {"node_id": "ERPcode/Catalogs/РесурсныеСпецификации",
                 "create_from_form": True, "evidence": "ANS.res_spec_yes"},
                {"node_id": "ERPcode/Catalogs/ЭтапыПроизводства",
                 "create_from_form": True, "evidence": "ANS.etap22_two_podr"},
                {"node_id": "ERPcode/Catalogs/Склады (Кладовая горячего цеха; Склад готовой продукции)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/Catalogs/Номенклатура (5 видов ГП, вид «Готовая продукция»)",
                 "create_from_form": True, "evidence": "block 7 output (5 видов)"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры (серия = дата выпечки + срок 5 дней)",
                 "create_from_form": True, "evidence": "ANS.serii_hleb_5_days"},
            ],
            "key_attributes": [
                "ЗаказНаПроизводство2_2 (основание)",
                "Предыдущий ЭтапПроизводства2_2 (холодный цех)",
                "Подразделение = 'Горячий цех'",
                "РесурснаяСпецификация (выпечка вида хлеба)",
                "ЭтапПроизводства (Формовка / Выпечка / Охлаждение)",
                "Склад получения полуфабриката = Кладовая горячего цеха",
                "Склад выпуска = Склад готовой продукции",
                "ТЧ «ВыходныеИзделия»: Номенклатура ГП (5 видов), Количество (шт.)",
                "ТЧ «Полуфабрикаты»: Номенклатура (Тесто), Количество (кг)",
                "ДатаНачала = ОкончаниеПредыдущегоЭтапа + 0.5 ч",
                "ДатаОкончания = Выпечка+Охлаждение",
                "Статус → 'Завершён'",
                "Серия = ДатаПроизводства (для контроля срока 5 дней)",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ОтборРазмещениеТоваров",
                 "relation": "Создать на основании (размещение на Складе ГП, ANS.sklsady → ОтборРазмещениеТоваров)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ЭтапПроизводства2_2",
                 "evidence": "dep_etap_22.json + ANS.tovary_na_skl"},
                {"register": "ERPcode/AccumulationRegisters/ДвиженияСерийТоваров",
                 "registrator": "ЭтапПроизводства2_2",
                 "evidence": "ANS.serii_systemic"},
                {"register": "ERPcode/AccumulationRegisters/ПартииПроизводства",
                 "registrator": "ЭтапПроизводства2_2",
                 "evidence": "audit GAP-8"},
            ],
            "controls": [
                "Контроль качества ГП (control:126, ANS.quality_control_owner)",
                "Контроль в 1С:ERP (ANS.kontrol_1c_erp)",
                "Электричество / мощность / транспорт — НЕ выделяются (constraint:128)",
                "Брак фиксируется (constraint:129)",
                "ФИФО (constraint:89, ANS.fifo_policy)",
                "Серия = Дата производства (на отчёт по срокам годности, ANS.serii_hleb_5_days)",
            ],
            "alt_branches": [
                {"condition": "Брак ГП",
                 "handling": "СписаниеТоваров + АктОСписанииТоваровУТМ (graph_gap на документ)"},
                {"condition": "Не хватает теста",
                 "handling": "Прерывание этапа + возврат на block 6.1"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ГП на Складе готовой продукции (по сериям)",
                "key_for_next": ["Запас ГП в шт. с серией и сроком годности"]
            },
        },
        {
            "step_id": "7.2",
            "name": "Размещение готовой продукции на Складе готовой продукции",
            "event": "ГП размещается по партиям (по дате производства) на СкладеГП",
            "primary_document": {
                "node_id": "ERPcode/Documents/ОтборРазмещениеТоваров",
                "title": "Документ Отбор (размещение) товаров",
                "evidence": "ANS.ordernaya_schema + ANS.sklsady = Склад ГП",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Склады (Склад готовой продукции)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры",
                 "create_from_form": False, "evidence": "перенос серии"},
            ],
            "key_attributes": [
                "Склад = Склад готовой продукции",
                "ТЧ «Товары»: Номенклатура ГП, Серия (Дата), Количество (шт.), Ячейка",
                "Дата размещения",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "РеализацияТоваровУслуг",
                 "relation": "Создать на основании (при отгрузке — block 10)"},
                {"created_doc": "ПеремещениеТоваров",
                 "relation": "Создать на основании (для собственного магазина — block 10)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ОтборРазмещениеТоваров",
                 "evidence": "dep_otbor.json"},
            ],
            "controls": [
                "Учёт остатков в штуках (control:136, ANS.units_only_shtuki)",
                "ФИФО (constraint:137, ANS.fifo_policy)",
                "Контроль остатков (control:143)",
            ],
            "alt_branches": [
                {"condition": "Срок хранения (5 дней) истекает",
                 "handling": "Блок 11 (Возвраты) или списание на утилизацию"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Запас ГП на Складе ГП (по сериям)",
                "key_for_next": ["Доступно для отгрузки клиентам/собственному магазину"]
            },
        },
    ],
    "outputs": [
        "Готовая продукция (5 видов: батон, хлеб белый в/с, бородинский, слойка, лепешка)",
        "Продукция передана на Склад готовой продукции",
    ]
}

block8 = {
    "block_id": 8,
    "title": "СКЛАДСКОЕ ХОЗЯЙСТВО (ХРАНЕНИЕ ГОТОВОЙ ПРОДУКЦИИ)",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:8",
    "scenario": {"id": None,
                 "title": None,
                 "layer": None,
                 "confidence": "graph_gap"},
    "summary": (
        "ГП хранится на СкладеГП в штуках, по партиям (дата производства). Срок "
        "хранения 5 дней. ФИФО. Учёт остатков через ТоварыНаСкладах и отчёт Остатки."
    ),
    "open_gaps": ["graph_gap: Сценарий 2.7 Планирование остатков неприменим для ГП "
                  "(ANS.min_zapas_pos_sk_period: ГП без лимита)"],
    "steps": [
        {
            "step_id": "8.1",
            "name": "Учёт остатков ГП и контроль сроков",
            "event": "Системный контроль сроков годности по сериям",
            "primary_document": {
                "node_id": "ERPcode/Reports/ОстаткиТоваровНаСкладах",
                "title": "Отчет Остатки товаров на складах",
                "evidence": "ANS.tovary_na_skl",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Склады (СкладГП)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры",
                 "create_from_form": False, "evidence": "ANS.serii_hleb_5_days"},
            ],
            "key_attributes": [
                "Отчётный период = каждый день (05:00 утра)",
                "Фильтр: Склад=СкладГП, Остаток>0",
                "Группировка по Номенклатуре + Серии (для прослеживания срока)",
                "Граница: Серии с истекающим 1-дневным сроком → красный маркер",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": None, "relation": "отчёт"},
            ],
            "registries_touched": [],
            "controls": [
                "Сроки хранения до 5 дней (control:142, ANS.serii_hleb_5_days)",
                "Контроль остатков (control:143)",
                "ФИФО (constraint:137, ANS.fifo_policy)",
                "Только штуки (ANS.units_only_shtuki)",
            ],
            "alt_branches": [
                {"condition": "Срок годности истёк (если ГП не ушла)",
                 "handling": "Блок 11 → возврат/списание/утилизация (control:217, ANS.c8f0bf1bcbe70621)"}
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Отчёт Остатки (оповещение мастеру смены)",
                "key_for_next": ["Список к списанию/утилизации"]
            },
        },
    ],
    "outputs": [
        "Учтенные остатки готовой продукции",
        "Готовая продукция к отгрузке",
    ]
}

block9 = {
    "block_id": 9,
    "title": "УПАКОВКА И ПОДГОТОВКА К ОТГРУЗКЕ",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:9",
    "scenario": {"id": None,
                 "title": None,
                 "layer": None,
                 "confidence": "graph_gap"},
    "summary": (
        "ГП подготавливается к отгрузке: часть упаковывается в потребительскую упаковку "
        "(Упаковки через справочник, ANS.hleb_units_upakovki), часть — в многооборотную "
        "тару (не используется, ANS.c7ed83841136dbe2). Формируются партии для отгрузки "
        "по ЗаказамКлиента, оформляется ОтборРазмещениеТоваров."
    ),
    "open_gaps": ["graph_gap: 9 строк с ТЧ «Упаковка» в ЗаказКлиента — уточнение реализации в L4"],
    "steps": [
        {
            "step_id": "9.1",
            "name": "Подготовка партий ГП по ЗаказамКлиента",
            "event": "Подбор ГП под каждый ЗаказКлиента (по серии, ФИФО)",
            "primary_document": {
                "node_id": "ERPcode/Documents/ОтборРазмещениеТоваров",
                "title": "Документ Отбор (размещение) товаров — отбор из СкладаГП в зону отгрузки",
                "evidence": "ANS.sklsady: СкладГП→ОтборРазмещение→Реализация; ANS.ordernaya_schema",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Склады (СкладГП, ЗонаОтгрузки)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/Catalogs/Упаковки",
                 "create_from_form": True, "evidence": "ANS.hleb_units_upakovki"},
                {"node_id": "ERPcode/Catalogs/Номенклатура (ГП + Упаковки как Характеристика)",
                 "create_from_form": False, "evidence": "ANS.hleb_units_upakovki (отдельная номенклатура — нет, через справочник Упаковки)"},
            ],
            "key_attributes": [
                "Склад = СкладГП",
                "Зона = ЗонаОтгрузки / ЗонаУпаковки",
                "ТЧ «Товары»: Номенклатура ГП, Упаковка (Упаковка→Коэффициент), КоличествоУпаковок, Количество (шт.), Серия",
                "Назначение = ЗаказКлиента",
                "Ответственный = Упаковщик",
                "Дата = Утро перед отгрузкой (перед утром послезавтра, по constraint:186)",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "РеализацияТоваровУслуг",
                 "relation": "Создать на основании → block 10"},
                {"created_doc": "ПеремещениеТоваров",
                 "relation": "Создать на основании → Склад собственного магазина (block 10)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыКОтгрузке",
                 "registrator": "ОтборРазмещениеТоваров",
                 "evidence": "dep_otbor.json + ANS.sklsady"},
                {"register": "ERPcode/AccumulationRegisters/РаспоряженияНаОтгрузкуИВозврат",
                 "registrator": "ОтборРазмещениеТоваров"},
            ],
            "controls": [
                "Проверка соответствия упаковки заказу (control:161)",
                "Проверка количества и качества упаковки (control:162)",
                "ФИФО (constraint:89, ANS.fifo_policy)",
                "Только штуки (ANS.units_only_shtuki) + Упаковки как справочник, не отдельная Номенклатура",
            ],
            "alt_branches": [
                {"condition": "Не хватает ГП в упаковке",
                 "handling": "Перепланирование отгрузки/производства (block 2)"},
                {"condition": "Многооборотная тара не используется",
                 "handling": "Этап 9 упрощается: упаковка только через справочник Упаковки (ANS.c7ed83841136dbe2)"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Партии ГП по заказам (готовы к отгрузке)",
                "key_for_next": ["Упакованная ГП в зоне отгрузки"]
            },
        },
    ],
    "outputs": [
        "Упакованная и подготовленная к отгрузке продукция",
        "Оформленные документы на отгрузку (отбор)",
    ]
}

block10 = {
    "block_id": 10,
    "title": "ОТГРУЗКА И ЛОГИСТИКА",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:10",
    "scenario": {"id": "scenario_005--6.5. Отгрузка товаров",
                 "title": "6.5. Отгрузка товаров",
                 "layer": 1,
                 "confidence": 0.5},
    "summary": (
        "Утро послезавтра. Крупные клиенты (сетевые магазины) забирают сами — ничего "
        "не оформляется (ANS.af96449542954017 = b). Для остальных партнёров и "
        "собственных магазинов доставка собственным транспортом. Внутреннее "
        "перемещение на Склад собственного магазина; Реализация ТУ для опт/сети."
    ),
    "open_gaps": ["graph_gap: документ ТранспортнаяНакладная — её отношение "
                  "к ЗаказНаДоставку для собственного магазина"],
    "steps": [
        {
            "step_id": "10.1",
            "name": "Самовывоз — для сетевых магазинов",
            "event": "Клиент приезжает, ему выдают ГП с зоны отгрузки",
            "primary_document": {
                "node_id": None,
                "title": "(документ не оформляется — самовывоз клиента)",
                "evidence": "ANS.af96449542954017 (b — ничего, клиент сам приезжает)",
            },
            "required_nsi": [],
            "key_attributes": [
                "Контроль отгрузки через отчёт «Движения по сериям/партиям» — физ. выдача фиксируется менеджером",
            ],
            "alternatives_create_on_basis": [],
            "registries_touched": [],
            "controls": [
                "Проверка комплектности отгрузки (control:182) — визуальный контроль + сверка с ЗаказомКлиента",
            ],
            "alt_branches": [
                {"condition": "Клиент не приехал",
                 "handling": "Оповещение менеджера, перепланирование (block 2 / переход в блок 11 → возврат)"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ГП отгружена (выдана) — без бумаги",
                "key_for_next": []
            },
        },
        {
            "step_id": "10.2",
            "name": "Отгрузка внешним клиентам (опт/сеть) — РеализацияТУ",
            "event": "Оформление РеализацияТоваровУслуг на основании ЗаказаКлиента",
            "primary_document": {
                "node_id": "ERPcode/Documents/РеализацияТоваровУслуг",
                "title": "Документ Реализация товаров и услуг",
                "ui_paths": [
                    "Продажи → ОптовыеПродажи → РеализацияТоваровУслуг",
                ],
                "evidence": "ANS.realiz_torg12_transportnaya; chain_klient_realiz.json",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Партнеры/Контрагенты",
                 "create_from_form": False, "evidence": "из ЗаказаКлиента"},
                {"node_id": "ERPcode/Catalogs/ДоговорыКонтрагентов",
                 "create_from_form": False, "evidence": "из ЗаказаКлиента"},
                {"node_id": "ERPcode/Catalogs/Склады (СкладГП)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/Catalogs/Номенклатура (ГП)",
                 "create_from_form": False, "evidence": "из ЗаказаКлиента"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры",
                 "create_from_form": False, "evidence": "ANS.serii_hleb_5_days"},
                {"node_id": "ERPcode/Documents/ТранспортнаяНакладная",
                 "create_from_form": True,
                 "evidence": "ANS.realiz_torg12_transportnaya (ТОРГ-12 + ТранспортнаяНакладная)"},
            ],
            "key_attributes": [
                "Основание = ЗаказКлиента",
                "Клиент (Партнер/Контрагент) — из ЗаказаКлиента",
                "Склад = СкладГП",
                "ТЧ «Товары»: Номенклатура ГП, Серия, Количество (шт.), Цена (из ВидаЦен)",
                "Печатные формы: ТОРГ-12, ТранспортнаяНакладная (если доставка)",
                "Статус = 'Отгружено'",
                "ХозяйственнаяОперация = Реализация",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ВозвратТоваровОтКлиента",
                 "relation": "Создать на основании (для будущего возврата — block 11)",
                 "evidence": "ANS.vozvrat_tov_ot_klienta + chain_realiz_vozvr.json"},
                {"created_doc": "ТранспортнаяНакладная",
                 "relation": "Создать на основании (если доставка)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "РеализацияТоваровУслуг (расход)",
                 "evidence": "dep_realiz.json + chain_klient_realiz.json"},
                {"register": "ERPcode/AccumulationRegisters/РасчетыСКлиентами",
                 "registrator": "РеализацияТоваровУслуг",
                 "evidence": "dep_realiz.json"},
                {"register": "ERPcode/AccumulationRegisters/ПартииТоваровОрганизаций",
                 "registrator": "РеализацияТоваровУслуг",
                 "evidence": "dep_realiz.json"},
                {"register": "ERPcode/AccumulationRegisters/Доставка",
                 "registrator": "РеализацияТоваровУслуг/ТранспортнаяНакладная",
                 "evidence": "dep_realiz.json"},
            ],
            "controls": [
                "Проверка комплектности отгрузки (control:182)",
                "Проверка оформления документов (control:183, отчёт КонтрольОформленияДокументовТовародвижений)",
                "Транспортные затраты не выделяются (constraint:185)",
                "ФИФО (constraint:89)",
                "Отгрузка согласно заявкам утро послезавтра (constraint:186)",
            ],
            "alt_branches": [
                {"condition": "Самовывоз",
                 "handling": "СпособДоставки='Самовывоз' в ЗаказеКлиента (ANS.samovyvoz_flag); бумага=ТОРГ-12 без ТранспортнойНакладной"},
                {"condition": "Доставка партнёру/опт",
                 "handling": "ANS.b7f6762afa63a891 — доставку опт/сеть делают сами"},
                {"condition": "Часть заказа не отгружена",
                 "handling": "Корректировка реализации / допоставка"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "РеализацияТоваровУслуг (статус «Отгружено»)",
                "key_for_next": ["Основание для будущего Возврата"]
            },
        },
        {
            "step_id": "10.3",
            "name": "Отгрузка в собственный магазин — Внутреннее перемещение + ЗаказНаДоставку",
            "event": "Перемещение ГП на Кладовую собственного магазина собственным транспортом",
            "primary_document": {
                "node_id": "ERPcode/Documents/ПеремещениеТоваров",
                "title": "Документ Перемещение товаров (внутреннее перемещение, q05)",
                "evidence": "ANS.q05_own_store (внутреннее перемещение, отдельный склад) + ANS.perem_tov_yes",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Склады (СкладГП → Кладовая собственного магазина)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed + ANS.zakaz_na_dostavku_sob"},
                {"node_id": "ERPcode/Documents/ЗаказНаДоставку",
                 "create_from_form": True,
                 "evidence": "ANS.zakaz_na_dostavku_sob (ЗаказНаДоставку для собственного магазина)"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры",
                 "create_from_form": False, "evidence": "ANS.serii_hleb_5_days"},
            ],
            "key_attributes": [
                "Основание = ЗаказКлиента (внутренний — для собственного магазина) + ЗаказНаДоставку",
                "Склад-отправитель = СкладГП",
                "Склад-получатель = Кладовая собственного магазина",
                "ТЧ «Товары»: Номенклатура ГП, Серия, Количество (шт.)",
                "Транспортное средство / Маршрут / Адрес доставки (магазина)",
                "Ответственный = Водитель / Экспедитор",
                "Статус Перемещения = 'Отгружено', получение = 'Принято кладовщиком магазина'",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "ВозвратТоваровОтКлиента",
                 "relation": "Создать на основании (для обратной приёмки — block 11)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ПеремещениеТоваров",
                 "evidence": "dep_perem.json"},
                {"register": "ERPcode/AccumulationRegisters/Доставка",
                 "registrator": "ЗаказНаДоставку / ПеремещениеТоваров",
                 "evidence": "dep_zakazdost.json"},
                {"register": "ERPcode/AccumulationRegisters/ДвиженияСерийТоваров",
                 "registrator": "ПеремещениеТоваров",
                 "evidence": "ANS.serii_systemic"},
            ],
            "controls": [
                "Проверка комплектности отгрузки (control:182)",
                "Проверка оформления документов (control:183, отчёт КонтрольОформленияДокументовТовародвижений)",
                "Транспортные затраты не выделяются (constraint:185)",
                "Отгрузка согласно заявкам утро послезавтра (constraint:186)",
            ],
            "alt_branches": [
                {"condition": "Магазин отменил заказ",
                 "handling": "Обратное ПеремещениеТоваров + ВозвратТоваровОтКлиента (между Организациями)"},
                {"condition": "Часть не доехала",
                 "handling": "АктОРасхождениях + довоз"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ПеремещениеТоваров (принято в магазине)",
                "key_for_next": ["ГП на Кладовой собственного магазина"]
            },
        },
    ],
    "outputs": [
        "Отгруженная продукция клиентам",
        "Оформленные документы: Реализация ТУ + ТОРГ-12 + ТранспортнаяНакладная",
        "ПеремещениеТоваров → Склад собственного магазина",
        "Зафиксированная выдача многооборотной тары (не используется — ANS.c7ed83841136dbe2)",
    ]
}

block11 = {
    "block_id": 11,
    "title": "ВОЗВРАТЫ ОТ КЛИЕНТОВ",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:11",
    "scenario": None,
    "summary": (
        "Клиент (или собственный магазин) оформляет ВозвратТоваровОтКлиента на основании "
        "РеализацияТоваровУслуг / ПеремещенияТоваров. Возвращённая продукция оценивается "
        "(свежая/просрочка). Утилизация возвратов НЕ отражается в учёте (ANS.515e341482fb0f00). "
        "Возврат денег — да, по решению (ANS.vozvrat_dengi)."
    ),
    "open_gaps": ["question:gap:bc161207b13359b5 — детализация возврата",
                  "question:bc161207b13359b5 — детали обработки возвратов (деньги / пересчёт)"],
    "steps": [
        {
            "step_id": "11.1",
            "name": "Оформление Возврата от клиента",
            "event": "Клиент инициирует возврат — оформляется ВозвратТоваровОтКлиента",
            "primary_document": {
                "node_id": "ERPcode/Documents/ВозвратТоваровОтКлиента",
                "title": "Документ Возврат товаров от клиента",
                "ui_paths": [
                    "Продажи → ОптовыеПродажи → ВозвратТоваровОтКлиента",
                ],
                "evidence": "ANS.vozvrat_tov_ot_klienta + process_plan.json operational_steps[3].ui_paths",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Партнеры/Контрагенты",
                 "create_from_form": False, "evidence": "из Реализации"},
                {"node_id": "ERPcode/Catalogs/ДоговорыКонтрагентов",
                 "create_from_form": False, "evidence": "из Реализации"},
                {"node_id": "ERPcode/Catalogs/Склады (Склад возврата/утилизации)",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
            ],
            "key_attributes": [
                "Основание = РеализацияТоваровУслуг (или ПеремещениеТоваров для собств. магазина)",
                "Партнер/Контрагент — из основания",
                "Склад = Склад возврата/утилизации (СкладГП — для свежей; отдельный — для просрочки)",
                "ТЧ «Товары»: Номенклатура ГП, Серия, Количество (шт.), Цена (обратная)",
                "ХозяйственнаяОперация = Возврат от клиента",
                "Возврат денег = Да (ANS.vozvrat_dengi + audit_q02)",
            ],
            "alternatives_create_on_basis": [
                {"created_doc": "АктОСписанииТоваровУТМ (утилизация)",
                 "relation": "Создать на основании (только для просрочки)",
                 "evidence": "graph_gap (см. блок gaps)"},
            ],
            "registries_touched": [
                {"register": "ERPcode/AccumulationRegisters/ТоварыНаСкладах",
                 "registrator": "ВозвратТоваровОтКлиента (приход)",
                 "evidence": "dep_vozvr.json + chain_realiz_vozvr.json"},
                {"register": "ERPcode/AccumulationRegisters/РасчетыСКлиентами",
                 "registrator": "ВозвратТоваровОтКлиента (сторнирование)",
                 "evidence": "dep_vozvr.json"},
                {"register": "ERPcode/AccumulationRegisters/ПартииТоваровОрганизаций",
                 "registrator": "ВозвратТоваровОтКлиента",
                 "evidence": "dep_vozvr.json"},
            ],
            "controls": [
                "Проверка состояния возвращённой продукции (control:200)",
                "Оформление возвратных документов (control:201)",
                "Утилизация не отражается в учёте (constraint:203, ANS.515e341482fb0f00)",
            ],
            "alt_branches": [
                {"condition": "Свежий хлеб — годен для продажи",
                 "handling": "Принять на СкладГП с серией 'возврат' (серия продолжает срок годности)"},
                {"condition": "Просроченный хлеб",
                 "handling": "Принять на отдельный Склад возврата/утилизации → списание (но учёт не отражает, ANS.515e341482fb0f00)"},
                {"condition": "Возврат по собств. магазину",
                 "handling": "ВозвратТоваровОтКлиента между Организациями (если магазин = отдельное ЮЛ) или обратное ПеремещениеТоваров"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "ВозвратТоваровОтКлиента (принят)",
                "key_for_next": ["Оприходование на склад возврата/ГП"]
            },
        },
        {
            "step_id": "11.2",
            "name": "Утилизация просроченной продукции (без учёта)",
            "event": "Списание просрочки (только описание действия; в учёте не отражается)",
            "primary_document": {
                "node_id": None,
                "title": "Действие (без документа в учёте)",
                "evidence": "constraint:203 + ANS.515e341482fb0f00 — утилизация не отражается",
            },
            "required_nsi": [],
            "key_attributes": [
                "Контроль отсутствия движений в ТоварыНаСкладах (через отчёт)",
                "Регистрация факта утилизации — только во внешней системе / журнале",
            ],
            "alternatives_create_on_basis": [],
            "registries_touched": [],
            "controls": [
                "Контроль сроков (constraint:211, ANS.kontrol_serii — через отчёт)",
                "Утилизация не в учёте (constraint:220, ANS.c8f0bf1bcbe70621)",
            ],
            "alt_branches": [
                {"condition": "Массовая просрочка (например, не вывезли с СГП)",
                 "handling": "ВозвратТоваровОтКлиента на себя + возвратное ПеремещениеТоваров в зону утилизации"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Журнал утилизации (вне ERP)",
                "key_for_next": []
            },
        },
    ],
    "outputs": [
        "Принятый возврат от клиента (ВозвратТоваровОтКлиента)",
        "Оформленные документы на возврат (Реализация сторнирована)",
        "Факт утилизации (без учёта)",
    ]
}

block12 = {
    "block_id": 12,
    "title": "УЧЕТ И ЕДИНИЦЫ ИЗМЕРЕНИЯ",
    "source_block_id": "task:подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f:block:12",
    "scenario": {"id": "scenario_006--14.6. Особенности методологии учета",
                 "title": "14.6. Особенности методологии учета",
                 "layer": 1,
                 "confidence": 0.571},
    "summary": (
        "Учёт ведётся только в оперативном+управленческом контуре "
        "(ИспользоватьРегламентированныйУчет=Ложь, ИспользоватьЗарплатаИКадры=Ложь). "
        "ГП — в штуках. ФИФО. Серии = сроки годности (особенно хлеб 5 дней). "
        "Отчёт: СебестоимостьТоваров, ОстаткиТоваровНаСкладах, КонтрольОформленияДокументов."
    ),
    "open_gaps": ["graph_gap: ИспользоватьУправленческийУчет / ИспользоватьРегламентированныйУчет — константы",
                  "graph_gap: точные метаданные отчётов (3 отчёта подтверждены в answers, "
                  "ночные имена отчётов — graph_gap; см. СебестоимостьТоваров ниже)"],
    "steps": [
        {
            "step_id": "12.1",
            "name": "Настройка учётной политики",
            "event": "Администратор включает оперативный/управленческий учёт и ФИФО",
            "primary_document": {
                "node_id": "ERPcode/Constants/ИспользоватьУправленческийУчет",
                "title": "Константа Использовать управленческий учет",
                "evidence": "ANS.c8f0bf1bcbe70621 (да — ИспользоватьЗарплатаИКадры=Ложь) + audit_q01 — граф не содержит прямой ссылки, "
                            "вывод по аналогии"},

            "required_nsi": [
                {"node_id": "ERPcode/Constants/ИспользоватьРегламентированныйУчет",
                 "create_from_form": False,
                 "evidence": "graph_gap (constraint:220 — нет бух. контура)"},
                {"node_id": "ERPcode/Constants/ИспользоватьЗарплатаИКадры",
                 "create_from_form": False,
                 "evidence": "ANS.c8f0bf1bcbe70621 (= Ложь)"},
                {"node_id": "ERPcode/Constants/ИспользоватьСерииНоменклатуры (для ГП и сырья)",
                 "create_from_form": False,
                 "evidence": "ANS.serii_systemic"},
            ],
            "key_attributes": [
                "ИспользоватьРегламентированныйУчет = Ложь (constraint:220)",
                "ИспользоватьУправленческийУчет = Истина (audit_q01)",
                "ИспользоватьЗарплатаИКадры = Ложь (ANS.c8f0bf1bcbe70621)",
                "МетодОценкиСтоимостиТоваров = ФИФО (ANS.fifo_policy)",
                "ИспользоватьСерии = Истина (ANS.serii_systemic)",
                "ИспользоватьКонтрольКачества = Истина (ANS.akt_kontrolya_kach)",
                "МинимальныйЗапасПоПозицииСкладуПериоду = Истина (ANS.min_zapas_pos_sk_period)",
                "ИспользоватьОрдернуюСхемуПриПриемке = Истина, ...ПриОтгрузке = Истина (ANS.ordernaya_schema)",
            ],
            "alternatives_create_on_basis": [],
            "registries_touched": [],
            "controls": [
                "Соответствие методологии ФИФО (control:217)",
                "Контроль сроков годности (control:218, ANS.kontrol_serii)",
            ],
            "alt_branches": [
                {"condition": "Включён бухгалтерский контур",
                 "handling": "Не должно происходить (constraint:220)"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Настройки учётной политики",
                "key_for_next": ["Все документы используют единую методологию"]
            },
        },
        {
            "step_id": "12.2",
            "name": "Получение отчётов по остаткам / движениям / себестоимости",
            "event": "Системное формирование отчётов для контроля",
            "primary_document": {
                "node_id": "ERPcode/Reports/ОстаткиТоваровНаСкладах",
                "title": "Отчет Остатки товаров на складах",
                "evidence": "ANS.tovary_na_skl (склад ГП, шт.)",
            },
            "required_nsi": [
                {"node_id": "ERPcode/Catalogs/Склады",
                 "create_from_form": False, "evidence": "ANS.all_sklsady_confirmed"},
                {"node_id": "ERPcode/InformationRegisters/СерииНоменклатуры",
                 "create_from_form": False, "evidence": "ANS.serii_hleb_5_days"},
            ],
            "key_attributes": [
                "Отчёт 1: «Остатки товаров на складах» — в шт.",
                "Отчёт 2: «Себестоимость товаров» — только в части материалов (audit_q01 + ANS.4c54af84321dafe5)",
                "Отчёт 3: «Движения товаров» — для списания (ANS.bf1b495ead686e9d)",
                "Отчёт 4: «Контроль оформления документов товародвижения» — для отслеживания неоформленных ордеров (control:183)",
                "Отчёт 5: «Анализ себестоимости продукции» — план/факт (graph_gap, есть только фрагментарно в сценарии 14.6)",
            ],
            "alternatives_create_on_basis": [],
            "registries_touched": [],
            "controls": [
                "Соответствие методологии ФИФО (control:217)",
                "Контроль сроков годности (control:218)",
            ],
            "alt_branches": [
                {"condition": "Расхождения остатков / себестоимости",
                 "handling": "Инвентаризация склада + корректировка регистров (graph_gap: документ ИнвентаризацияРасхождений)"},
            ],
            "artifacts_for_next_step": {
                "primary_doc": "Отчётность для менеджера / руководства",
                "key_for_next": []
            },
        },
    ],
    "outputs": [
        "Актуальные данные по остаткам, движению продукции, себестоимости (в части материалов)",
    ]
}

# -----------------------------------------------------------------
# Final plan structure
# -----------------------------------------------------------------

PROCESS_PLAN = {
    "$schema": "RAGAgent.agent_process_plan.v1",
    "task_id": "подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f",
    "scenario": "Хлебопекарное производство под заказ (Make-to-Order): 5 видов хлеба, 1 полуфабрикат (Тесто), 2 цеха (холодный/горячий), 6 складов, доставка утро послезавтра, FIFO, серии сроков годности (хлеб 5 дней), возврат + утилизация без учёта, без бух. контура",
    "trigger_event": "Заявка от клиента (до 12:00) → ЗаказКлиента (статус «К производству»)",
    "horizon": "сегодня (T) → заказ на T+1 (производство) → отгрузка утром T+2 (послезавтра)",
    "graph_evidence_sources": [
        "agent_context.json (12 process blocks)",
        "answers.json (60 confirmed decisions)",
        "process_plan.json operational_steps (graph-only normalization)",
        "_tmp_dep/dep_*.json (dependencies, 12 documents): incl. inline-candidate NSI list",
        "_tmp_dep/chain_*.json (document-chain ЗаказКлиента→РеализацияТУ; "
        "ЗаказПоставщику→Приобретение; ЗаказНаПроизводство2_2→ЭтапПроизводства2_2; "
        "РеализацияТУ→ВозвратТоваровОтКлиента; АктКонтроляКачестваТоваров→Приобретение)",
        "query_graph.py metadata hits",
    ],
    "ui_path_vocabulary": [
        "Продажи → ОптовыеПродажи → ЗаказКлиента",
        "Продажи → ОптовыеПродажи → РеализацияТоваровУслуг",
        "Продажи → ОптовыеПродажи → ВозвратТоваровОтКлиента",
        "Закупки → Закупки → ЗаказПоставщику",
        "Закупки → Закупки → ПриобретениеТоваровУслуг",
        "Производство → МежцеховоеУправление2_2 → ЗаказНаПроизводство2_2",
        "Производство → МежцеховоеУправление2_2 → ЭтапыПроизводства2_2",
        "БюджетированиеИПланирование → ПланированиеЗапасов → ПланПроизводства",
        "Склад и доставка → Управление доставкой → Заказ на доставку",
    ],
    "sklady_confirmed_by_user": [
        "Склад сырья",
        "Кладовая холодного цеха",
        "Кладовая горячего цеха",
        "Склад готовой продукции (СкладГП)",
        "Склад собственного магазина (Кладовая собственного магазина)",
        "Склад возврата / утилизации",
    ],
    "blocks": [block1, block2, block3, block4, block5,
               block6, block7, block8, block9, block10, block11, block12],
    "confirmed_decisions_reference": {
        "answers_json_path": f"{BASE}/answers.json (60 confirmed answers)",
        "stop_clarification_at": 60,
        "open_assumptions": [
            "Контур учета = оперативный + управленческий, без бухгалтерского (ИспользоватьРегламентированныйУчет=Ложь)",
            "5 видов ГП — отдельные элементы в Номенклатура (вид «Готовая продукция»)",
            "Полуфабрикат Тесто — отдельная Номенклатура (вид «Полуфабрикат») с собственной ресурсной спецификацией",
            "Учёт по партиям включён (учётная политика)",
            "Ценообразование: единый прайс + соглашения для опт/сеть/собственный магазин; "
            "для собственного магазина используется внутренняя себестоимостная цена",
            "Доставку опт/сеть делают сами (самoвывоз); собственный магазин — мы сами, "
            "внутреннее перемещение на своём транспорте через ЗаказНаДоставку",
            "Минимальные лимиты только для расходников и материалов (не для ГП)",
            "Ордерная схема включена везде (приёмка, перемещение, отгрузка)",
            "ФИФО — в учётной политике",
            "Серии — хлеб 5 дней, мука — стандартный срок годности (произв. серии)",
            "Возврат денег клиенту при возврате товара",
            "Брак фиксируется; утилизация в учёте не отражается (только описание)",
            "Зарплата во внешнем приложении (ИспользоватьЗарплатаИКадры=Ложь)",
            "Многооборотная тара не используется",
        ]
    },
    "open_gaps_summary": [],
    "graph_gaps_summary": [],
}

# Collect open_gaps and graph_gaps from blocks
all_open = set()
all_gaps = set()
for b in PROCESS_PLAN["blocks"]:
    for s in b.get("steps", []):
        for ab in s.get("alt_branches", []):
            if ab.get("graph_gap"):
                all_gaps.add(ab["graph_gap"])
    for og in b.get("open_gaps", []):
        all_open.add(og)

# additional graph_gaps discovered in step primary_documents or alt paths
for b in PROCESS_PLAN["blocks"]:
    for s in b["steps"]:
        for ab in s.get("alternatives_create_on_basis", []):
            ev = ab.get("evidence", "")
            if "graph_gap" in ev:
                all_gaps.add(ev)

PROCESS_PLAN["open_gaps_summary"] = sorted(all_open)
PROCESS_PLAN["graph_gaps_summary"] = sorted(all_gaps)

OUT = f"{BASE}/agent_process_plan.json"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(PROCESS_PLAN, f, ensure_ascii=False, indent=2, default=str)

print(f"OK -> {OUT}")
print(f"blocks: {len(PROCESS_PLAN['blocks'])}")
print(f"open_gaps_summary: {PROCESS_PLAN['open_gaps_summary']}")
print(f"graph_gaps_summary: {PROCESS_PLAN['graph_gaps_summary']}")
