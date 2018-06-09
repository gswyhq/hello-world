#!/usr/bin/python3
# coding: utf-8

# 将新问题写入数据库
import traceback
import pymysql.err
from datetime import datetime
from logger.logger import logger
# from conf.web_qa import NEW_QUESTION_TABLE_CONF
from mysql_connect.mysql_conn import new_question_mysql, IntegrityError
# from mysql_connect.mysql_create_table import CreateTable
from conf.mysql_db import NEW_QUESTION_TABLE_NAME, QUESTION_TABLE_NAME, ENTITY_TABLE_NAME
from dialogue_managem import global_variable
from conf.nwd_conf import QA_FILE, QA_FILE_SPLIT

from auxiliary.extract_datetime import datetime_match_pattern

class NewQuestion():
    def __init__(self, db=new_question_mysql.db, table_name=NEW_QUESTION_TABLE_NAME):
        self.db=db
        self.table_name = table_name

    def insert_one_record(self, uid='', question='', answer='', **kwargs):
        """
        向新问题数据库中，插入一条新问题
        :param uid:
        :param question:
        :param answer:
        :param kwargs:
        :return:
        """
        try:
            if not self.db:
                logger.info("没有主动链接MySQL数据库")
                return False
            self.db.ping()  # 检测数据连接是否正常
            field_list = 'question, answer, status, created, updated'
            value_list = ' "{}", "{}", 0, NOW(), NOW()'.format(question, answer)
            sql = 'INSERT INTO {} ( {} ) VALUES ( {} )'.format(self.table_name, field_list, value_list)
            with self.db.cursor() as cursor:
                try:
                    cursor.execute(sql)
                    self.db.commit()
                    return True
                except IntegrityError:
                    logger.info("问题已存在, 更新数据")
                    sql = 'UPDATE {} SET answer="{}", updated=NOW() WHERE question="{}";'.format(self.table_name, answer, question)
                    cursor.execute(sql)
                    self.db.commit()
                    return True
        except Exception as e:
            logger.info("出错：{}".format(e))
            logger.info("错误详情：{}".format(traceback.format_exc()))

        return False

    def question_insert_one_record(self, uid='', question='', answer='', similar_question='', tag1=0, tag2=0, **kwargs):
        """
        向用户配置的数据库中，插入一条新问题
        :param uid:
        :param question:
        :param answer:
        :param kwargs:
        :return:
        """
        table_name = 'question'
        try:
            if not self.db:
                logger.info("没有主动链接MySQL数据库")
                return False
            self.db.ping()  # 检测数据连接是否正常
            field_list = 'question, answer, similar_question, tag1, tag2, created, updated, status'
            value_list = ' "{}", "{}", "{}", "{}", "{}", NOW(), NOW(), 1 '.format(question, answer, similar_question, tag1, tag2)
            sql = 'INSERT INTO {} ( {} ) VALUES ( {} )'.format(table_name, field_list, value_list)
            with self.db.cursor() as cursor:
                try:
                    cursor.execute(sql)
                    self.db.commit()
                    return True
                except IntegrityError:
                    logger.info("问题已存在")
        except Exception as e:
            logger.info("出错：{}".format(e))
            logger.info("错误详情：{}".format(traceback.format_exc()))

        return False

    def question_insert_executemany_record(self, uid, data, tag1=0, tag2=0, status=1, **kwargs):
        """
        批量向用户配置的问题库中添加配置问题
        :param uid:
        :param data:
        :param tag1:
        :param tag2:
        :param status:
        :param kwargs:
        :return:
        """

        table_name = 'question'
        try:
            if not self.db:
                logger.info("没有主动链接MySQL数据库")
                return False
            self.db.ping()  # 检测数据连接是否正常
            field_list = 'question, answer, similar_question, tag1, tag2, created, updated, status'
            value_list = []
            for d in data:
                question = d.get('question')
                answer = d.get('answer')
                if question and answer:
                    similar_question = d.get('similar_question', '')
                    value = (question, answer, similar_question, tag1, tag2, datetime.now(), datetime.now(), status)
                    value_list.append(value)
                else:
                    continue

            sql = 'INSERT INTO {} ( {} ) VALUES ( {} )'.format(table_name, field_list, '%s, %s, %s, %s, %s, %s, %s, %s')
            with self.db.cursor() as cursor:
                try:
                    cursor.executemany(sql, value_list)
                    self.db.commit()
                    return True
                except IntegrityError:
                    logger.info("问题已存在")
        except Exception as e:
            logger.info("出错：{}".format(e))
            logger.info("错误详情：{}".format(traceback.format_exc()))

        return False

    def entity_insert_executemany_record(self, entity_file='/home/gswewf/yhb/input/同义词导入模板.xls', status=1, **kwargs):
        """
        向MySQL数据库中，写入同义词
        :param uid: 
        :param data: 
        :param status: 
        :param kwargs: 
        :return: 
        """
        import xlrd
        # 打开文件
        workbook = xlrd.open_workbook(entity_file)
        sheet1 = workbook.sheet_by_index(0)  # sheet索引从0开始
        nrow = sheet1.nrows  # 行数
        # ncol = sheet1.ncols  # 列数
        data = []
        for row in range(nrow):
            text = sheet1.cell_value(row, 0)
            if text == '标准词':
                continue
            word = sheet1.cell_value(row, 0)
            similar_word = sheet1.cell_value(row, 1) or ''
            similar_word = similar_word.replace('，', ',')
            data.append({"word": word, "similar_word": similar_word})
        ret = self.import_entity(data, status=status, **kwargs)
        return ret

    def import_entity(self, data, status=1, **kwargs):
        """
        向MySQL数据库导入同义词
        :param data: 
        :return: 
        """
        table_name = 'entity'
        try:
            if not self.db:
                logger.info("没有主动链接MySQL数据库")
                return False
            self.db.ping()  # 检测数据连接是否正常
            field_list = 'word, similar_word, created, updated, status'
            value_list = []
            for d in data:
                word = d.get('word')
                similar_word = d.get('similar_word', '')
                if word:
                    value = (word, similar_word, datetime.now(), datetime.now(), status)
                    value_list.append(value)
                else:
                    continue

            sql = 'INSERT INTO {} ( {} ) VALUES ( {} )'.format(table_name, field_list, '%s, %s, %s, %s, %s')
            with self.db.cursor() as cursor:
                try:
                    cursor.executemany(sql, value_list)
                    self.db.commit()
                    return True
                except IntegrityError:
                    logger.info("问题已存在")
        except Exception as e:
            logger.info("出错：{}".format(e))
            logger.info("错误详情：{}".format(traceback.format_exc()))

        return False

    def gen_sql(self, question):
        """生成查询语句
        当提到天气时，且没有说到时间，就查询今天的；若说的时间是今天的，也查今天的；若说的时间不是今天，则直接搜索
        """

        if "天气" in question:
            # 图灵查天气，不管说查什么时候的天气，都是返回当天的天气；
            # datetime_result = datetime_match_pattern(sentence=question)
            # if datetime_result[0]:
            #     parse_datetime = datetime_result[1]
            #     diff_days = (parse_datetime - datetime.now()).days + 1
            # else:
            #     logger.info("没有解析出时间，则认为是查询今天的天气")
            #     diff_days = 1
            # today_sql = 'SELECT *, DATEDIFF(NOW(), updated) as "diff" FROM {table_name} WHERE {question_filed} like "%{question}%" HAVING diff < {diff_days};'.format(
            #         question=question,
            #         table_name=self.table_name, question_filed='question', diff_days=diff_days)
            today_sql = 'SELECT *, DATEDIFF(NOW(), updated) as "diff" FROM {table_name} WHERE {question_filed} like "%{question}%" HAVING diff < {diff_days};'.format(
                    question=question,
                    table_name=self.table_name, question_filed='question', diff_days=1)
            return today_sql

        sql = 'SELECT * FROM {table_name}  WHERE {question_filed} like "%{question}%"'.format(question=question,
                                                                                                  table_name=self.table_name,
                                                                                                  question_filed='question')
        return sql

    def search_question(self, question=''):
        """
        在问题库中查询一个问题
        :param question:
        :return:
        """
        try:
            if not self.db:
                logger.info("没有主动链接MySQL数据库")
                return {}
            try:
                self.db.close()  # 必须要强行关闭数据库，不然查询的数据不是最新的
            except pymysql.err.Error:
                logger.info("数据库已经关闭")
            self.db.ping()  # 检测数据连接是否正常
            sql = self.gen_sql(question)
            # logger.info("sql: {}".format(sql))
            with self.db.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchone() or {}
        except Exception as e:
            logger.info("出错：{}".format(e))
            logger.info("错误详情：{}".format(traceback.format_exc()))
            result = {}
        logger.info("问题`{}`的查询结果： {}".format(question, result))
        return result

    def dump_all_question_answer(self):
        """
        导出一对一问答库中所有的问题,及导出同义词库中的同义词表
        :return:
        """
        try:
            if not self.db:
                logger.info("没有主动链接MySQL数据库")
                return {}
            try:
                self.db.close()  # 必须要强行关闭数据库，不然查询的数据不是最新的
            except pymysql.err.Error:
                logger.info("数据库已经关闭")
            self.db.ping()  # 检测数据连接是否正常
            with self.db.cursor() as cursor:
                question_table_name = QUESTION_TABLE_NAME
                sql1 = "select id, question, answer, similar_question, tag1, tag2 from {table_name} WHERE status=1".format(table_name=question_table_name)
                cursor.execute(sql1)
                question_answer = cursor.fetchall() or []
                question_answer = self.replace_split(question_answer)
                # with self.db.cursor() as cursor:
                logger.info("从数据库{}，表： {}， 读取了{}条一对一问答".format(new_question_mysql.get_host, question_table_name, len(question_answer)))

                entity_table_name = ENTITY_TABLE_NAME
                sql2 = "select word, similar_word from {table_name} WHERE status=1".format(table_name=entity_table_name)
                cursor.execute(sql2)
                entity = cursor.fetchall() or []
                logger.info("从数据库{}, 表： {}， 读取了{}条同义词".format(new_question_mysql.get_host, entity_table_name, len(entity)))
        except Exception as e:
            logger.info("查询出错：{}".format(e))
            logger.info("错误详情：{}".format(traceback.format_exc()))
            question_answer = []
            entity = []
        return {"question_answer": question_answer, "entity": entity}

    def replace_split(self, question_answer):
        """将相似问题中的分隔符由`$`替换成`$$`"""
        new_question_answer = []

        for data in question_answer:
            similar_question = data.get('similar_question', '')
            if '$' in similar_question and '$$' not in similar_question:
                similar_question = similar_question.replace('$', '$$')
                data['similar_question'] = similar_question
            new_question_answer.append(data)
        return new_question_answer

class TxtQuestion(NewQuestion):
    def __init__(self):
        NewQuestion.__init__(self, db=new_question_mysql.db, table_name=NEW_QUESTION_TABLE_NAME)

    def dump_all_question_answer(self, qa_file=QA_FILE):
        with open(qa_file, encoding='utf8')as f:
            qa_datas = f.readlines()

        entity = []
        question_answer = []
        for line in qa_datas:
            line = line.strip()

            line_split = line.split(QA_FILE_SPLIT)
            assert len(line_split) >= 2, "问答数据格式不符合要求： {}".format(line)

            question = line_split[0]
            answer = line_split[-1]
            similar_question = line_split[1:-1]

            data = {
                "question": question,
                "answer": answer,
                "similar_question": QA_FILE_SPLIT.join(similar_question)
            }
            question_answer.append(data)

        all_data = {
            "question_answer": question_answer,
            "entity": entity
        }
        return all_data

    def load_to_mysql(self, qa_file=QA_FILE):
        """
        从文件中将用户配置的问答导入到数据库中
        :param qa_file:
        :return:
        """
        with open(qa_file, encoding='utf8')as f:
            qa_datas = f.readlines()


        question_answer = []
        for line in qa_datas:
            line = line.strip()

            line_split = line.split(QA_FILE_SPLIT)
            assert len(line_split) >= 2, "问答数据格式不符合要求： {}".format(line)

            question = line_split[0]
            answer = line_split[-1]
            similar_question = line_split[1:-1]

            data = {
                "question": question,
                "answer": answer,
                "similar_question": QA_FILE_SPLIT.join(similar_question)
            }
            question_answer.append(data)
        print(question_answer)
        return self.question_insert_executemany_record(uid='123456', data=question_answer)


if global_variable.QA_SOURCE == 'txt':
    new_question = TxtQuestion()
else:
    new_question = NewQuestion()

def main():
    # new_question.create_new_question_table()  # 创建新问题数据表
    # result = new_question.insert_one_record(uid='123', question='吃饭没有', answer='我不知道')  # 向数据库插入一条新问题
    # print(result)
    # result = new_question.search_question(question="今天天气")
    # print(result)

    dumps_data = new_question.dump_all_question_answer()
    print(dumps_data)
    # print(len(dumps_data), type(dumps_data))

    # print(dumps_data.get('question_answer')[:3])
    # print(dumps_data.get('entity')[:3])

    # new_question.load_to_mysql()

if __name__ == '__main__':
    main()