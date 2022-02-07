 
import java.awt.List;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;

import org.postgresql.jdbc2.ArrayAssistantRegistry;

//<dependency>
//<groupId>org.postgresql</groupId>
//<artifactId>postgresql</artifactId>
//<version>42.2.2</version>
//</dependency>

public class PostgreSqlJdbcConnSelectDatas {
	/*创建 PostgreSQL表；*/
	public static String createTable() {
	      Connection c = null;
	      Statement stmt = null;
	      try {
	         Class.forName("org.postgresql.Driver");
	         c = DriverManager
	            .getConnection("jdbc:postgresql://localhost:5432/db_person",
	            "username", "password");
	         System.out.println("连接数据库成功！");
	         stmt = c.createStatement();
	         String sql = "CREATE TABLE COMPANY02 " +
                     "(ID INT PRIMARY KEY     NOT NULL," +
                     " NAME           TEXT    NOT NULL, " +
                     " AGE            INT     NOT NULL, " +
                     " ADDRESS        CHAR(50), " +
                     " SALARY         REAL)";
	         stmt.executeUpdate(sql);
	         stmt.close();
	         c.close();
	         
	      } catch (Exception e) {
	         e.printStackTrace();
	         System.err.println(e.getClass().getName()+": "+e.getMessage());
	         System.exit(0);
	      }
	      System.out.println("新表创建成功！");
	      return "ok";
	   }

	/* 查询 postgresql 数据；*/
	public static String readData() {
		Connection c = null;
		Statement stmt = null;
		try {
			Class.forName("org.postgresql.Driver");
			c = DriverManager.getConnection(
					"jdbc:postgresql://localhost:5432/db_person", "username",
					"password");
			c.setAutoCommit(false);
 
			System.out.println("连接数据库成功！");
			stmt = c.createStatement();
			String queryURL = String.format("SELECT * FROM pg_db.pg_table1 where guid='%s'", "94648475-32f7-40cc-1935-cc732ad6aee1");
			ResultSet rs = stmt.executeQuery(queryURL);
			while(rs.next()){
				int id = rs.getInt("id");
				String guid = rs.getString("guid");
				String db = rs.getString("db");
				String table = rs.getString("table_collision");
				int colType = rs.getInt("col_type");
				System.out.println(id + "," + guid + "," + db + "," + table.trim() + "," + colType);
			}
 
			rs.close();
			stmt.close();
			
			c.close();
 
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getClass().getName() + ": " + e.getMessage());
			System.exit(0);
		}
		System.out.println("查询数据成功！");
		return "ok";
	}
	
	/*向postgresql插入数据；*/
	public static void insertData() {
		Connection c = null;
		Statement stmt = null;
		try {
			Class.forName("org.postgresql.Driver");
			c = DriverManager.getConnection(
					"jdbc:postgresql://localhost:5432/db_person", "username",
					"password");
			c.setAutoCommit(false);
 
			System.out.println("连接数据库成功！");
			stmt = c.createStatement();
 
			String sql = "INSERT INTO pg_db.pg_table1 (guid, db, table_collision, column_collision, col_type, from_collision, to_collision, saturation, entropy, heat, scene, source_collision, tag, desc_collision, created_date, updated_date) "
					+ "VALUES ( '11111111-c2be-1a91-a1bf-2e327e05c2c5', 'abc_change_db', 'client_info', 'top_3_city', '0', '{\"guidEntityMap\": {\"53485959-c2be-1a91-a1bf-2e327e05c2c5\": {\"db\": \"abc_change_db\", \"table\": \"client_info\", \"column\": \"top_3_city\", \"scene\": \"互联网金融\", \"source\": \"bank\", \"tag\": \"位置标签\", \"desc\": \"整体最高频活动城市TOP3\", \"saturation\": \"0.97\", \"entropy\": \"19.1\", \"heat\": \"3\"}, \"0f53a426-3c05-f923-cca8-b744c0ba2a97\": {\"db\": \"abc_change_db\", \"table\": \"client_info\", \"column\": \"top_3_city\", \"scene\": \"互联网金融\", \"source\": \"bank\", \"tag\": \"位置标签\", \"desc\": \"整体最高频活动城市TOP3\", \"saturation\": \"0.47\", \"entropy\": \"18.1\", \"heat\": \"6\"}, \"8ac84c30-47e7-99a4-0c89-c2c6f6c45f5c\": {\"db\": \"abc_change_db\", \"table\": \"client_info_p\", \"column\": \"top_3_city\", \"scene\": \"互联网金融\", \"source\": \"bank\", \"tag\": \"位置标签\", \"desc\": \"整体最高频活动城市TOP3\", \"saturation\": \"0.26\", \"entropy\": \"17.8\", \"heat\": \"6\"}, \"82123376-f70e-7e39-b89d-57052db6722a\": {\"db\": \"abc_change_db\", \"table\": \"client_data_collision_info\", \"column\": \"top_3_city\", \"scene\": \"互联网金融\", \"source\": \"bank\", \"tag\": \"位置标签\", \"desc\": \"整体最高频活动城市TOP3\", \"saturation\": \"0.32\", \"entropy\": \"16.9\", \"heat\": \"7\"}}, \"relations\": [{\"fromEntityId\": \"53485959-c2be-1a91-a1bf-2e327e05c2c5\", \"toEntityId\": \"0f53a426-3c05-f923-cca8-b744c0ba2a97\"}, {\"fromEntityId\": \"0f53a426-3c05-f923-cca8-b744c0ba2a97\", \"toEntityId\": \"8ac84c30-47e7-99a4-0c89-c2c6f6c45f5c\"}, {\"fromEntityId\": \"8ac84c30-47e7-99a4-0c89-c2c6f6c45f5c\", \"toEntityId\": \"82123376-f70e-7e39-b89d-57052db6722a\"}]}', '{\"guidEntityMap\": {\"53485959-c2be-1a91-a1bf-2e327e05c2c5\": {\"db\": \"abc_change_db\", \"table\": \"client_info\", \"column\": \"top_3_city\", \"scene\": \"互联网金融\", \"source\": \"bank\", \"tag\": \"位置标签\", \"desc\": \"整体最高频活动城市TOP3\", \"saturation\": \"0.97\", \"entropy\": \"19.1\", \"heat\": \"3\"}, \"0f53a426-3c05-f923-cca8-b744c0ba2a97\": {\"db\": \"abc_change_db\", \"table\": \"client_info\", \"column\": \"top_3_city\", \"scene\": \"互联网金融\", \"source\": \"bank\", \"tag\": \"位置标签\", \"desc\": \"整体最高频活动城市TOP3\", \"saturation\": \"0.47\", \"entropy\": \"18.1\", \"heat\": \"6\"}, \"8ac84c30-47e7-99a4-0c89-c2c6f6c45f5c\": {\"db\": \"abc_change_db\", \"table\": \"client_info_p\", \"column\": \"top_3_city\", \"scene\": \"互联网金融\", \"source\": \"bank\", \"tag\": \"位置标签\", \"desc\": \"整体最高频活动城市TOP3\", \"saturation\": \"0.26\", \"entropy\": \"17.8\", \"heat\": \"6\"}, \"82123376-f70e-7e39-b89d-57052db6722a\": {\"db\": \"abc_change_db\", \"table\": \"client_data_collision_info\", \"column\": \"top_3_city\", \"scene\": \"互联网金融\", \"source\": \"bank\", \"tag\": \"位置标签\", \"desc\": \"整体最高频活动城市TOP3\", \"saturation\": \"0.32\", \"entropy\": \"16.9\", \"heat\": \"7\"}}, \"relations\": [{\"fromEntityId\": \"53485959-c2be-1a91-a1bf-2e327e05c2c5\", \"toEntityId\": \"0f53a426-3c05-f923-cca8-b744c0ba2a97\"}, {\"fromEntityId\": \"0f53a426-3c05-f923-cca8-b744c0ba2a97\", \"toEntityId\": \"8ac84c30-47e7-99a4-0c89-c2c6f6c45f5c\"}, {\"fromEntityId\": \"8ac84c30-47e7-99a4-0c89-c2c6f6c45f5c\", \"toEntityId\": \"82123376-f70e-7e39-b89d-57052db6722a\"}]}', '0.97', '19.1', '3', '互联网金融', 'bank', '位置标签', '整体最高频活动城市TOP3', '2021-03-26 09:24:25', '2021-03-26 09:24:25' );";
			stmt.executeUpdate(sql);
 
			stmt.close();
			c.commit();
			c.close();
 
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getClass().getName() + ": " + e.getMessage());
			System.exit(0);
		}
		System.out.println("新增数据成功！");
	}
	
	/*删除PostgreSQL数据；*/
	public static String deleteData() {
		Connection c = null;
		Statement stmt = null;
		try {
			Class.forName("org.postgresql.Driver");
			c = DriverManager.getConnection(
					"jdbc:postgresql://localhost:5432/db_person", "username",
					"password");
			c.setAutoCommit(false);
 
			System.out.println("连接数据库成功！");
			stmt = c.createStatement();
			
			String countSql = "SELECT count(*) rec FROM pg_db.pg_table1";
			ResultSet rc1 = stmt.executeQuery(countSql);
			int rowCount = 0;
			if(rc1.next()) {
				rowCount=rc1.getInt("rec");
			}
			System.out.println(String.format("删除数据记录前，总共含有记录数：%s", rowCount));
			rc1.close();
			c.commit();
			
			String sql = "Delete from pg_db.pg_table1 where guid='11111111-c2be-1a91-a1bf-2e327e05c2c5' ";
			Integer rc2 = stmt.executeUpdate(sql);
			System.out.println(String.format("删除数据记录返回结果：%s", rc2));
			c.commit();
 
			ResultSet rc3 = stmt.executeQuery(countSql);
			int rowCount2 = 0;
			if(rc3.next()) {
				rowCount2=rc3.getInt("rec");
			}
			System.out.println(String.format("删除数据记录后，总共含有记录数：%s", rowCount2));
			rc3.close();
			c.commit();
			
			stmt.close();
			
			c.close();
 
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getClass().getName() + ": " + e.getMessage());
			System.exit(0);
		}
		System.out.println("删除更新数据成功！");
		return "ok";
	}

	/*更新 postgreSQL数据；*/
	public static String updateData() {
		Connection c = null;
		Statement stmt = null;
		try {
			Class.forName("org.postgresql.Driver");
			c = DriverManager.getConnection(
					"jdbc:postgresql://localhost:5432/db_person", "username",
					"password");
			c.setAutoCommit(false);
 
			System.out.println("连接数据库成功！");
			stmt = c.createStatement();
			
			// 查询旧数据；
			String query_id = "2958";
			String queryURL = String.format("SELECT * FROM pg_db.pg_table1 where id='%s'", query_id);
			ResultSet rs1 = stmt.executeQuery(queryURL);
			while(rs1.next()){
				int id = rs1.getInt("id");
				String guid = rs1.getString("guid");
				System.out.println(String.format("更新前，id 为 %s 的记录，对应guid为：%s", id, guid));
			}
			rs1.close();
			c.commit();
			
			// 更新数据；
			String sql = String.format("UPDATE pg_db.pg_table1 set guid = '24' where id='%s' ", query_id);
			Integer rs2 = stmt.executeUpdate(sql);
			System.out.println(String.format("更新pg数据的结果 %s", rs2));
			c.commit();
 
			ResultSet rs3 = stmt.executeQuery(queryURL);
			while(rs3.next()){
				int id = rs3.getInt("id");
				String guid = rs3.getString("guid");
				System.out.println(String.format("更新后，id 为 %s 的记录，对应guid为：%s", id, guid));
			}
 
			rs3.close();
			stmt.close();
			
			c.close();
 
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getClass().getName() + ": " + e.getMessage());
			System.exit(0);
		}
		System.out.println("更新数据成功！");
		return "ok";
	}

	public static void main(String args[]) {
//		createTable(); // 创建表
//		String retString = readData();  // 查询数据；
//		System.out.println(retString);
		insertData(); // 插入数据；
//		updateData(); // 更新数据；
		deleteData(); // 删除数据；
	}
	
}




