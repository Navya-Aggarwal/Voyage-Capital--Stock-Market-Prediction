CREATE DATABASE IF NOT EXISTS `projectlogin` DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
USE `projectlogin`;

CREATE TABLE IF NOT EXISTS `userdetails`(
	 `id` int(11) NOT NULL AUTO_INCREMENT PRIMARY KEY,
	 `email` varchar(50) NOT NULL,
     `username` varchar(30) UNIQUE NOT NULL,
     `password` varchar(20) NOT NULL,
     `firstname` varchar(50) NOT NULL,
     `lastname` varchar(50) NOT NULL,
     `mobile` bigint(10) NOT NULL,
     `address` varchar(200) NOT NULL
);

CREATE TABLE IF NOT EXISTS `userstock`(
     `id` int(9) NOT NULL AUTO_INCREMENT PRIMARY KEY,
     `stock1` char(10) NOT NULL,
     `stock2` char(10) NOT NULL,
     `stock3` char(10) NOT NULL,
     foreign key (id) references userdetails(id)
);

select * from userdetails;
select * from userstock;

CREATE TABLE IF NOT EXISTS `messages`(
     `messageid` int(9) NOT NULL AUTO_INCREMENT PRIMARY KEY,
     `fname` varchar(50) NOT NULL,
     `email` varchar(30) NOT NULL,
     `message` mediumtext NOT NULL
);

select * from messages;