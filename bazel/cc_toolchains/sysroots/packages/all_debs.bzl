load("//bazel/rules_pkg:pkg_provider_archives.bzl", "deb_archive_w_pkg_providers")

def deb_repos():
    deb_archive_w_pkg_providers(
        name = "debian12_aardvark-dns_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "790c8cfcd867b0bc31b3e0feb6a05b3c938a2f67219a02ada0bcaf7afc768b1c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/790c8cfcd867b0bc31b3e0feb6a05b3c938a2f67219a02ada0bcaf7afc768b1c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/aardvark-dns/aardvark-dns_1.4.0-3_arm64.deb"],
        deps = ["@debian12_libgcc-s1_aarch64//:all_files", "@debian12_netavark_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_aardvark-dns_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e60c3f659a014fab3afa76fcd9776dbc77bd108f43d1c48b9b66585d7c0d4bc8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e60c3f659a014fab3afa76fcd9776dbc77bd108f43d1c48b9b66585d7c0d4bc8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/aardvark-dns/aardvark-dns_1.4.0-3_amd64.deb"],
        deps = ["@debian12_libgcc-s1_x86_64//:all_files", "@debian12_netavark_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_adduser_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c24fe4eb8e60d8632d72ed104cce7c92cff200847c897dc8ba764b6c47b519e0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c24fe4eb8e60d8632d72ed104cce7c92cff200847c897dc8ba764b6c47b519e0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/adduser/adduser_3.134_all.deb"],
        deps = ["@debian12_passwd_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_adduser_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c24fe4eb8e60d8632d72ed104cce7c92cff200847c897dc8ba764b6c47b519e0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c24fe4eb8e60d8632d72ed104cce7c92cff200847c897dc8ba764b6c47b519e0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/adduser/adduser_3.134_all.deb"],
        deps = ["@debian12_passwd_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_base-files_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2c7151f66540a3fa0fa2727675c82de4906d26fb0fe985fb4af4f555ff7d90ee",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2c7151f66540a3fa0fa2727675c82de4906d26fb0fe985fb4af4f555ff7d90ee.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/b/base-files/base-files_12.4+deb12u3_arm64.deb"],
        deps = ["@debian12_gawk_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_base-files_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "7327b2aff303fa1ab2dec0ed34f814013c311d3513bbf3578106f5088356f2b2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7327b2aff303fa1ab2dec0ed34f814013c311d3513bbf3578106f5088356f2b2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/b/base-files/base-files_12.4+deb12u3_amd64.deb"],
        deps = ["@debian12_gawk_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_bash_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "13c4e70030a059aeec6b745e4ce2949ce67405246bb38521e6c8f4d21c133543",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/13c4e70030a059aeec6b745e4ce2949ce67405246bb38521e6c8f4d21c133543.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/b/bash/bash_5.2.15-2+b2_arm64.deb"],
        deps = ["@debian12_base-files_aarch64//:all_files", "@debian12_debianutils_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_bash_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5325e63acaecb37f6636990328370774995bd9b3dce10abd0366c8a06877bd0d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5325e63acaecb37f6636990328370774995bd9b3dce10abd0366c8a06877bd0d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/b/bash/bash_5.2.15-2+b2_amd64.deb"],
        deps = ["@debian12_base-files_x86_64//:all_files", "@debian12_debianutils_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_ca-certificates_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5308b9bd88eebe2a48be3168cb3d87677aaec5da9c63ad0cf561a29b8219115c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5308b9bd88eebe2a48be3168cb3d87677aaec5da9c63ad0cf561a29b8219115c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/ca-certificates/ca-certificates_20230311_all.deb"],
        deps = ["@debian12_debconf_aarch64//:all_files", "@debian12_openssl_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_ca-certificates_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5308b9bd88eebe2a48be3168cb3d87677aaec5da9c63ad0cf561a29b8219115c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5308b9bd88eebe2a48be3168cb3d87677aaec5da9c63ad0cf561a29b8219115c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/ca-certificates/ca-certificates_20230311_all.deb"],
        deps = ["@debian12_debconf_x86_64//:all_files", "@debian12_openssl_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_conmon_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9d15cab56dc01b228c6d889b8159c1a45eb995c5938cbc9467bc19e01f36e3e7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9d15cab56dc01b228c6d889b8159c1a45eb995c5938cbc9467bc19e01f36e3e7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/conmon/conmon_2.1.6+ds1-1_arm64.deb"],
        deps = ["@debian12_libglib2.0-0_aarch64//:all_files", "@debian12_libsystemd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_conmon_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2d4111853199dd8c7cbcda3da04f37d9b7f3a7b08dc1cee546a1e5518de8a596",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2d4111853199dd8c7cbcda3da04f37d9b7f3a7b08dc1cee546a1e5518de8a596.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/conmon/conmon_2.1.6+ds1-1_amd64.deb"],
        deps = ["@debian12_libglib2.0-0_x86_64//:all_files", "@debian12_libsystemd0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_coreutils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ec8f090a14c684879dce251254d8d9ed0876d4480f750d5807ef04e5435e1c4d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ec8f090a14c684879dce251254d8d9ed0876d4480f750d5807ef04e5435e1c4d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/coreutils/coreutils_9.1-1_arm64.deb"],
        deps = ["@debian12_libacl1_aarch64//:all_files", "@debian12_libattr1_aarch64//:all_files", "@debian12_libgmp10_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_coreutils_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "61038f857e346e8500adf53a2a0a20859f4d3a3b51570cc876b153a2d51a3091",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/61038f857e346e8500adf53a2a0a20859f4d3a3b51570cc876b153a2d51a3091.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/coreutils/coreutils_9.1-1_amd64.deb"],
        deps = ["@debian12_libacl1_x86_64//:all_files", "@debian12_libattr1_x86_64//:all_files", "@debian12_libgmp10_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_crun_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e4a27bf7e75746e9d9500985402e83eb9c9d7742135ff038f0bd729a3cde2c20",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e4a27bf7e75746e9d9500985402e83eb9c9d7742135ff038f0bd729a3cde2c20.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/crun/crun_1.8.1-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libcap2_aarch64//:all_files", "@debian12_libsystemd0_aarch64//:all_files", "@debian12_libyajl2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_crun_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "34bdcf1a097ec34da2e7b77b65fa351134702af775174af22555b16421709c51",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/34bdcf1a097ec34da2e7b77b65fa351134702af775174af22555b16421709c51.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/crun/crun_1.8.1-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libcap2_x86_64//:all_files", "@debian12_libsystemd0_x86_64//:all_files", "@debian12_libyajl2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_dash_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c1358e2a8054eb93efd460adf480224a16ea9e0b4d7b4c6cbcf8c8c91902a1d7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c1358e2a8054eb93efd460adf480224a16ea9e0b4d7b4c6cbcf8c8c91902a1d7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/dash/dash_0.5.12-2_arm64.deb"],
        deps = ["@debian12_debianutils_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_dash_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "33ea40061da2f1a861ec46212b2b6a34f0776a049b1a3f0abce2fb8cb994258f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/33ea40061da2f1a861ec46212b2b6a34f0776a049b1a3f0abce2fb8cb994258f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/dash/dash_0.5.12-2_amd64.deb"],
        deps = ["@debian12_debianutils_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_debconf_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "74ab14194a3762b2fc717917dcfda42929ab98e3c59295a063344dc551cd7cc8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/74ab14194a3762b2fc717917dcfda42929ab98e3c59295a063344dc551cd7cc8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/debconf/debconf_1.5.82_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_debconf_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "74ab14194a3762b2fc717917dcfda42929ab98e3c59295a063344dc551cd7cc8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/74ab14194a3762b2fc717917dcfda42929ab98e3c59295a063344dc551cd7cc8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/debconf/debconf_1.5.82_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_debianutils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "0fc03c548293aee2359af53bd03b30ab42ca3493afcb65bed2f3caee90ffd46a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0fc03c548293aee2359af53bd03b30ab42ca3493afcb65bed2f3caee90ffd46a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/debianutils/debianutils_5.7-0.5~deb12u1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_debianutils_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "55f951359670eb3236c9e2ccd5fac9ccb3db734f5a22aff21589e7a30aee48c9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/55f951359670eb3236c9e2ccd5fac9ccb3db734f5a22aff21589e7a30aee48c9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/debianutils/debianutils_5.7-0.5~deb12u1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_dirmngr_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1fe11b28d8ea206c5a5a71b843731c3d4f2e1a671c13a3be7de5f80f29893d43",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1fe11b28d8ea206c5a5a71b843731c3d4f2e1a671c13a3be7de5f80f29893d43.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/dirmngr_2.2.40-1.1_arm64.deb"],
        deps = ["@debian12_adduser_aarch64//:all_files", "@debian12_gpgconf_aarch64//:all_files", "@debian12_init-system-helpers_aarch64//:all_files", "@debian12_libassuan0_aarch64//:all_files", "@debian12_libgcrypt20_aarch64//:all_files", "@debian12_libgnutls30_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files", "@debian12_libksba8_aarch64//:all_files", "@debian12_libldap-2.5-0_aarch64//:all_files", "@debian12_libnpth0_aarch64//:all_files", "@debian12_lsb-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_dirmngr_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e3a8e56057592c60fd8db174968e9f232f07905b79544a9e477cd48f008326b2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e3a8e56057592c60fd8db174968e9f232f07905b79544a9e477cd48f008326b2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/dirmngr_2.2.40-1.1_amd64.deb"],
        deps = ["@debian12_adduser_x86_64//:all_files", "@debian12_gpgconf_x86_64//:all_files", "@debian12_init-system-helpers_x86_64//:all_files", "@debian12_libassuan0_x86_64//:all_files", "@debian12_libgcrypt20_x86_64//:all_files", "@debian12_libgnutls30_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files", "@debian12_libksba8_x86_64//:all_files", "@debian12_libldap-2.5-0_x86_64//:all_files", "@debian12_libnpth0_x86_64//:all_files", "@debian12_lsb-base_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_dmsetup_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5c838de946282ee64c8483886c8bc68276d3fc5a9a28ce23bbfa6577fe0ddac2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5c838de946282ee64c8483886c8bc68276d3fc5a9a28ce23bbfa6577fe0ddac2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lvm2/dmsetup_1.02.185-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_dmsetup_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c73fc490b93c83550ed272de69ec96c5da30d4456b889f9e93c7fd8e53860b85",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c73fc490b93c83550ed272de69ec96c5da30d4456b889f9e93c7fd8e53860b85.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lvm2/dmsetup_1.02.185-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_dpkg_aarch64",
        sha256 = "4775b5f27315054c6860f402d1ef74614090e61a4192ec9c63cec9c88a75f9c2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4775b5f27315054c6860f402d1ef74614090e61a4192ec9c63cec9c88a75f9c2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/dpkg/dpkg_1.21.22_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libmd0_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files", "@debian12_libzstd1_aarch64//:all_files", "@debian12_tar_aarch64//:all_files", "@debian12_zlib1g_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_dpkg_x86_64",
        sha256 = "9d97f27d8a8a06dd4800e8e0291337ca02e11cdfd7df09a4566a982a6d9fe4c4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9d97f27d8a8a06dd4800e8e0291337ca02e11cdfd7df09a4566a982a6d9fe4c4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/dpkg/dpkg_1.21.22_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libmd0_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files", "@debian12_libzstd1_x86_64//:all_files", "@debian12_tar_x86_64//:all_files", "@debian12_zlib1g_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gawk_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "38f1cfc73847954dd2a0a94e414472a0d6ea0d1bb0c399d1f6e1948e15ff5880",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/38f1cfc73847954dd2a0a94e414472a0d6ea0d1bb0c399d1f6e1948e15ff5880.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gawk/gawk_5.2.1-2_arm64.deb"],
        deps = ["@debian12_libgmp10_aarch64//:all_files", "@debian12_libmpfr6_aarch64//:all_files", "@debian12_libreadline8_aarch64//:all_files", "@debian12_libsigsegv2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gawk_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9cd63c1b35ff082092c221a23dcb167f72c4d1c3de3a42e11f16181f42ab3b55",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9cd63c1b35ff082092c221a23dcb167f72c4d1c3de3a42e11f16181f42ab3b55.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gawk/gawk_5.2.1-2_amd64.deb"],
        deps = ["@debian12_libgmp10_x86_64//:all_files", "@debian12_libmpfr6_x86_64//:all_files", "@debian12_libreadline8_x86_64//:all_files", "@debian12_libsigsegv2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gcc-12-base_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e1f2fb7212546c0e360af8df26303608f7b09e123ac9c96e15872d1ec1ce3275",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e1f2fb7212546c0e360af8df26303608f7b09e123ac9c96e15872d1ec1ce3275.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/gcc-12-base_12.2.0-14_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gcc-12-base_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1a03df5a57833d65b5bb08cfa19d50e76f29088dc9e64fb934af42d9023a0807",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1a03df5a57833d65b5bb08cfa19d50e76f29088dc9e64fb934af42d9023a0807.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/gcc-12-base_12.2.0-14_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gnupg-l10n_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "bc62f3b366042157e9a8d00d04f1bd2e2a05e37501fc9a821883f99aa282ed77",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bc62f3b366042157e9a8d00d04f1bd2e2a05e37501fc9a821883f99aa282ed77.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gnupg-l10n_2.2.40-1.1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gnupg-l10n_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "bc62f3b366042157e9a8d00d04f1bd2e2a05e37501fc9a821883f99aa282ed77",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bc62f3b366042157e9a8d00d04f1bd2e2a05e37501fc9a821883f99aa282ed77.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gnupg-l10n_2.2.40-1.1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gnupg-utils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "cad0666b9d862301241add8ea1e789beab39fe966471add38a3bf33a29769fa9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cad0666b9d862301241add8ea1e789beab39fe966471add38a3bf33a29769fa9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gnupg-utils_2.2.40-1.1_arm64.deb"],
        deps = ["@debian12_libassuan0_aarch64//:all_files", "@debian12_libbz2-1.0_aarch64//:all_files", "@debian12_libgcrypt20_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files", "@debian12_libksba8_aarch64//:all_files", "@debian12_libreadline8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gnupg-utils_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6156f5b9edc0de38755869e5bcbed0b65d48d2a5531ae2f0ff2c347a7882f402",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6156f5b9edc0de38755869e5bcbed0b65d48d2a5531ae2f0ff2c347a7882f402.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gnupg-utils_2.2.40-1.1_amd64.deb"],
        deps = ["@debian12_libassuan0_x86_64//:all_files", "@debian12_libbz2-1.0_x86_64//:all_files", "@debian12_libgcrypt20_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files", "@debian12_libksba8_x86_64//:all_files", "@debian12_libreadline8_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gnupg_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6f6fe95c43338db9887e52fe948228a779d3651fef1a975b62dfe891bb71fdc4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6f6fe95c43338db9887e52fe948228a779d3651fef1a975b62dfe891bb71fdc4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gnupg_2.2.40-1.1_all.deb"],
        deps = ["@debian12_dirmngr_aarch64//:all_files", "@debian12_gnupg-l10n_aarch64//:all_files", "@debian12_gnupg-utils_aarch64//:all_files", "@debian12_gpg-agent_aarch64//:all_files", "@debian12_gpg-wks-client_aarch64//:all_files", "@debian12_gpg-wks-server_aarch64//:all_files", "@debian12_gpg_aarch64//:all_files", "@debian12_gpgsm_aarch64//:all_files", "@debian12_gpgv_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gnupg_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6f6fe95c43338db9887e52fe948228a779d3651fef1a975b62dfe891bb71fdc4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6f6fe95c43338db9887e52fe948228a779d3651fef1a975b62dfe891bb71fdc4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gnupg_2.2.40-1.1_all.deb"],
        deps = ["@debian12_dirmngr_x86_64//:all_files", "@debian12_gnupg-l10n_x86_64//:all_files", "@debian12_gnupg-utils_x86_64//:all_files", "@debian12_gpg-agent_x86_64//:all_files", "@debian12_gpg-wks-client_x86_64//:all_files", "@debian12_gpg-wks-server_x86_64//:all_files", "@debian12_gpg_x86_64//:all_files", "@debian12_gpgsm_x86_64//:all_files", "@debian12_gpgv_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_golang-github-containers-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "39d5c31902c4ce55bd419286946d0b291383169585f1b3509000f6bf57454181",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/39d5c31902c4ce55bd419286946d0b291383169585f1b3509000f6bf57454181.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/golang-github-containers-common/golang-github-containers-common_0.50.1+ds1-4_all.deb"],
        deps = ["@debian12_golang-github-containers-image_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_golang-github-containers-common_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "39d5c31902c4ce55bd419286946d0b291383169585f1b3509000f6bf57454181",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/39d5c31902c4ce55bd419286946d0b291383169585f1b3509000f6bf57454181.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/golang-github-containers-common/golang-github-containers-common_0.50.1+ds1-4_all.deb"],
        deps = ["@debian12_golang-github-containers-image_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_golang-github-containers-image_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2dc5697a30ae8de43221eeb7aa553d5187dbf97392d299ecb27ef51e46d159ca",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2dc5697a30ae8de43221eeb7aa553d5187dbf97392d299ecb27ef51e46d159ca.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/golang-github-containers-image/golang-github-containers-image_5.23.1-4_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_golang-github-containers-image_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2dc5697a30ae8de43221eeb7aa553d5187dbf97392d299ecb27ef51e46d159ca",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2dc5697a30ae8de43221eeb7aa553d5187dbf97392d299ecb27ef51e46d159ca.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/golang-github-containers-image/golang-github-containers-image_5.23.1-4_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpg-agent_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a30073d7f66be59d267e35c92d40ac79ddab7e6f3a83a88a9824c7cd572c89f1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a30073d7f66be59d267e35c92d40ac79ddab7e6f3a83a88a9824c7cd572c89f1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpg-agent_2.2.40-1.1_arm64.deb"],
        deps = ["@debian12_gpgconf_aarch64//:all_files", "@debian12_init-system-helpers_aarch64//:all_files", "@debian12_libassuan0_aarch64//:all_files", "@debian12_libgcrypt20_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files", "@debian12_libnpth0_aarch64//:all_files", "@debian12_pinentry-curses_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpg-agent_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ac48d6bfac9298843355561a14047673a9361ecff7f24cfe1da119dbf1a037e9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ac48d6bfac9298843355561a14047673a9361ecff7f24cfe1da119dbf1a037e9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpg-agent_2.2.40-1.1_amd64.deb"],
        deps = ["@debian12_gpgconf_x86_64//:all_files", "@debian12_init-system-helpers_x86_64//:all_files", "@debian12_libassuan0_x86_64//:all_files", "@debian12_libgcrypt20_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files", "@debian12_libnpth0_x86_64//:all_files", "@debian12_pinentry-curses_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpg-wks-client_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2d29ba3b4f2aae5fd8dc58c8068e95049586a53bb9da264114a1280687184da7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2d29ba3b4f2aae5fd8dc58c8068e95049586a53bb9da264114a1280687184da7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpg-wks-client_2.2.40-1.1_arm64.deb"],
        deps = ["@debian12_libassuan0_aarch64//:all_files", "@debian12_libgcrypt20_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpg-wks-client_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2520093a31c082ace185a18ad6bdf860b13f32139977d1dfe1d52867c2e5df30",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2520093a31c082ace185a18ad6bdf860b13f32139977d1dfe1d52867c2e5df30.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpg-wks-client_2.2.40-1.1_amd64.deb"],
        deps = ["@debian12_libassuan0_x86_64//:all_files", "@debian12_libgcrypt20_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpg-wks-server_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "da9a9ae1b996535bdf96a8f92605f8a18178f67c8f5d9e9d905a3d00ee4fe124",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/da9a9ae1b996535bdf96a8f92605f8a18178f67c8f5d9e9d905a3d00ee4fe124.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpg-wks-server_2.2.40-1.1_arm64.deb"],
        deps = ["@debian12_libgcrypt20_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpg-wks-server_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "7516082b33a0e3c76d6c18d67754d5f2ef2116255fac9897ff0eb2004aa8de8c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7516082b33a0e3c76d6c18d67754d5f2ef2116255fac9897ff0eb2004aa8de8c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpg-wks-server_2.2.40-1.1_amd64.deb"],
        deps = ["@debian12_libgcrypt20_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpg_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "047e5e0bb65df5acc487cb8feafb5b6bea5e96d63c037d4979d988a8a1db14ec",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/047e5e0bb65df5acc487cb8feafb5b6bea5e96d63c037d4979d988a8a1db14ec.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpg_2.2.40-1.1_arm64.deb"],
        deps = ["@debian12_gpgconf_aarch64//:all_files", "@debian12_libassuan0_aarch64//:all_files", "@debian12_libbz2-1.0_aarch64//:all_files", "@debian12_libgcrypt20_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files", "@debian12_libreadline8_aarch64//:all_files", "@debian12_libsqlite3-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpg_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d66fd8d7dd21a98e6a5acaa8d3fcb80b30561bb20c8e635dd6e66873abd4d40d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d66fd8d7dd21a98e6a5acaa8d3fcb80b30561bb20c8e635dd6e66873abd4d40d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpg_2.2.40-1.1_amd64.deb"],
        deps = ["@debian12_gpgconf_x86_64//:all_files", "@debian12_libassuan0_x86_64//:all_files", "@debian12_libbz2-1.0_x86_64//:all_files", "@debian12_libgcrypt20_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files", "@debian12_libreadline8_x86_64//:all_files", "@debian12_libsqlite3-0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpgconf_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "23ed984a2b2e5818b19878358b9bb9aab70bb7833745b95840703e3e96f56258",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/23ed984a2b2e5818b19878358b9bb9aab70bb7833745b95840703e3e96f56258.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpgconf_2.2.40-1.1_arm64.deb"],
        deps = ["@debian12_libassuan0_aarch64//:all_files", "@debian12_libgcrypt20_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files", "@debian12_libreadline8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpgconf_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b3a0cc418526e1f9ae90ed320714cbdcf28dc252e7b5dddbf885cbe4062b3c63",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b3a0cc418526e1f9ae90ed320714cbdcf28dc252e7b5dddbf885cbe4062b3c63.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpgconf_2.2.40-1.1_amd64.deb"],
        deps = ["@debian12_libassuan0_x86_64//:all_files", "@debian12_libgcrypt20_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files", "@debian12_libreadline8_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpgsm_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9b5f2633eb0857a0651a0d5f8f2e165107996dd145ef4f0af8c539a195f75bea",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9b5f2633eb0857a0651a0d5f8f2e165107996dd145ef4f0af8c539a195f75bea.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpgsm_2.2.40-1.1_arm64.deb"],
        deps = ["@debian12_gpgconf_aarch64//:all_files", "@debian12_libassuan0_aarch64//:all_files", "@debian12_libgcrypt20_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files", "@debian12_libksba8_aarch64//:all_files", "@debian12_libreadline8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpgsm_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "37d5e8d44bb9729a89d747db15880f0f01e53101cc16f258087bb8b591017e76",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/37d5e8d44bb9729a89d747db15880f0f01e53101cc16f258087bb8b591017e76.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpgsm_2.2.40-1.1_amd64.deb"],
        deps = ["@debian12_gpgconf_x86_64//:all_files", "@debian12_libassuan0_x86_64//:all_files", "@debian12_libgcrypt20_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files", "@debian12_libksba8_x86_64//:all_files", "@debian12_libreadline8_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpgv_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ed98c000084fc38ae9d2e0e33cb7f76fc573aa91ff623330eec5cffb5de40ccb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ed98c000084fc38ae9d2e0e33cb7f76fc573aa91ff623330eec5cffb5de40ccb.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpgv_2.2.40-1.1_arm64.deb"],
        deps = ["@debian12_libbz2-1.0_aarch64//:all_files", "@debian12_libgcrypt20_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_gpgv_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "0a43a9785f32d517a967d99e00d8e0a69edc0be09d4e63a08d7fd64466a11a0f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0a43a9785f32d517a967d99e00d8e0a69edc0be09d4e63a08d7fd64466a11a0f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnupg2/gpgv_2.2.40-1.1_amd64.deb"],
        deps = ["@debian12_libbz2-1.0_x86_64//:all_files", "@debian12_libgcrypt20_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_grep_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2caf08393ef6ea18a52e7e4609a7bd13811ff53733b4e45d2c7b0957e8d592cf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2caf08393ef6ea18a52e7e4609a7bd13811ff53733b4e45d2c7b0957e8d592cf.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/grep/grep_3.8-5_arm64.deb"],
        deps = ["@debian12_libpcre2-8-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_grep_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3264acea728df3c48a54f20e9291b965130e306b9d00adac76647049da7196df",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3264acea728df3c48a54f20e9291b965130e306b9d00adac76647049da7196df.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/grep/grep_3.8-5_amd64.deb"],
        deps = ["@debian12_libpcre2-8-0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_icu-devtools_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "94a2230a3f1526a75ff9d822b37c3cb3c328b07e6c308ff202a51f9aa247fd91",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/94a2230a3f1526a75ff9d822b37c3cb3c328b07e6c308ff202a51f9aa247fd91.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/icu/icu-devtools_72.1-3_arm64.deb"],
        deps = ["@debian12_libgcc-s1_aarch64//:all_files", "@debian12_libicu72_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_icu-devtools_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "da6c6ca530353acefca95ed82852313df1eca7f0296b905ced738be4f129b7d9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/da6c6ca530353acefca95ed82852313df1eca7f0296b905ced738be4f129b7d9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/icu/icu-devtools_72.1-3_amd64.deb"],
        deps = ["@debian12_libgcc-s1_x86_64//:all_files", "@debian12_libicu72_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_init-system-helpers_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f9ce24cbf69957dc1851fc55adba0a60b5bc617d51587b6478f2be64786442f1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f9ce24cbf69957dc1851fc55adba0a60b5bc617d51587b6478f2be64786442f1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/init-system-helpers/init-system-helpers_1.65.2_all.deb"],
        deps = ["@debian12_usrmerge_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_init-system-helpers_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f9ce24cbf69957dc1851fc55adba0a60b5bc617d51587b6478f2be64786442f1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f9ce24cbf69957dc1851fc55adba0a60b5bc617d51587b6478f2be64786442f1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/init-system-helpers/init-system-helpers_1.65.2_all.deb"],
        deps = ["@debian12_usrmerge_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_iptables_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5cb1974950a07f89c2d61a20e62d2dbcd882acb6d522d36ad3c690f6f5d3c643",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5cb1974950a07f89c2d61a20e62d2dbcd882acb6d522d36ad3c690f6f5d3c643.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/iptables/iptables_1.8.9-2_arm64.deb"],
        deps = ["@debian12_libip4tc2_aarch64//:all_files", "@debian12_libip6tc2_aarch64//:all_files", "@debian12_libmnl0_aarch64//:all_files", "@debian12_libnetfilter-conntrack3_aarch64//:all_files", "@debian12_libnfnetlink0_aarch64//:all_files", "@debian12_libnftnl11_aarch64//:all_files", "@debian12_libxtables12_aarch64//:all_files", "@debian12_netbase_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_iptables_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2928d3ca6ca8a3dc3f423f6752822b1f3614a5ba609ff7806bcba4449ffe90e1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2928d3ca6ca8a3dc3f423f6752822b1f3614a5ba609ff7806bcba4449ffe90e1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/iptables/iptables_1.8.9-2_amd64.deb"],
        deps = ["@debian12_libip4tc2_x86_64//:all_files", "@debian12_libip6tc2_x86_64//:all_files", "@debian12_libmnl0_x86_64//:all_files", "@debian12_libnetfilter-conntrack3_x86_64//:all_files", "@debian12_libnfnetlink0_x86_64//:all_files", "@debian12_libnftnl11_x86_64//:all_files", "@debian12_libxtables12_x86_64//:all_files", "@debian12_netbase_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libacl1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2b0eef11a2e271e7355adaf1f6cbf8d2e83835ae1b6cf15165d59b8289c08342",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2b0eef11a2e271e7355adaf1f6cbf8d2e83835ae1b6cf15165d59b8289c08342.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/acl/libacl1_2.3.1-3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libacl1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8be9df5795114bfe90e2be3d208ef47a5edd3fc7b3e20d387a597486d444e5e2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8be9df5795114bfe90e2be3d208ef47a5edd3fc7b3e20d387a597486d444e5e2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/acl/libacl1_2.3.1-3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libasan8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "7dbf07af4c6a5bb53e31f7e487c1c65f3168959cd4015d66f8d0fec660289a94",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7dbf07af4c6a5bb53e31f7e487c1c65f3168959cd4015d66f8d0fec660289a94.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libasan8_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files", "@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libasan8_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "30b4972cc88a4ff0fba9e08e6d476de13b109af9e4b826d130bdc72771d6e373",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/30b4972cc88a4ff0fba9e08e6d476de13b109af9e4b826d130bdc72771d6e373.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libasan8_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files", "@debian12_libgcc-s1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libassuan0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f74048f54d824401b9aa4bb26fcabe1e4a29cd36f1331ceab8b474d5af59f24f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f74048f54d824401b9aa4bb26fcabe1e4a29cd36f1331ceab8b474d5af59f24f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/liba/libassuan/libassuan0_2.5.5-5_arm64.deb"],
        deps = ["@debian12_libgpg-error0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libassuan0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5912430927da16ccc831459679207fdbb9dfc5a206f2bab8d6f36d5a1ab53e25",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5912430927da16ccc831459679207fdbb9dfc5a206f2bab8d6f36d5a1ab53e25.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/liba/libassuan/libassuan0_2.5.5-5_amd64.deb"],
        deps = ["@debian12_libgpg-error0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libatomic1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "25a32fd0fe8b083fea31d2819d50e7254a0b9e529477c0740bbb44cbe297ec70",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/25a32fd0fe8b083fea31d2819d50e7254a0b9e529477c0740bbb44cbe297ec70.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libatomic1_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libatomic1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a35f744972476c4b425e006d5c0752d917f3a6f48ce1268723a29e65a65b78a6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a35f744972476c4b425e006d5c0752d917f3a6f48ce1268723a29e65a65b78a6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libatomic1_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libattr1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "481e1c3fcad6773ba1c9d7f1de59e4fc80d95326c9b20192e13b7111013c932a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/481e1c3fcad6773ba1c9d7f1de59e4fc80d95326c9b20192e13b7111013c932a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/attr/libattr1_2.5.1-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libattr1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c4945123d66d0503ba42e2fc0585abc76d0838978c6d277b9cc37a4da25d1a34",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c4945123d66d0503ba42e2fc0585abc76d0838978c6d277b9cc37a4da25d1a34.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/attr/libattr1_2.5.1-4_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libaudit-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "17d0341ca6ce604ce59c296780ac2c2a24141a769823c50669af942c025e6591",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/17d0341ca6ce604ce59c296780ac2c2a24141a769823c50669af942c025e6591.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/audit/libaudit-common_3.0.9-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libaudit-common_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "17d0341ca6ce604ce59c296780ac2c2a24141a769823c50669af942c025e6591",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/17d0341ca6ce604ce59c296780ac2c2a24141a769823c50669af942c025e6591.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/audit/libaudit-common_3.0.9-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libaudit1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "619606427a334cba955e0afb18bf4a636df4141d32ea474a79cc512b5ca358e7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/619606427a334cba955e0afb18bf4a636df4141d32ea474a79cc512b5ca358e7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/audit/libaudit1_3.0.9-1_arm64.deb"],
        deps = ["@debian12_libaudit-common_aarch64//:all_files", "@debian12_libcap-ng0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libaudit1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "30954df4b5a7c505661ba8ae5e6ea94f5805e408899fb400783bb166eb5ff306",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/30954df4b5a7c505661ba8ae5e6ea94f5805e408899fb400783bb166eb5ff306.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/a/audit/libaudit1_3.0.9-1_amd64.deb"],
        deps = ["@debian12_libaudit-common_x86_64//:all_files", "@debian12_libcap-ng0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libblkid1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "23d4db3a890310bd3a10370f4104a9618f0b92830625434799da6f85e7a6dbd6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/23d4db3a890310bd3a10370f4104a9618f0b92830625434799da6f85e7a6dbd6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/u/util-linux/libblkid1_2.38.1-5+b1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libblkid1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "146ee93768433ac6a33edc8ae9248d8d619f10ef42c18b1212e0cb594ab9be3b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/146ee93768433ac6a33edc8ae9248d8d619f10ef42c18b1212e0cb594ab9be3b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/u/util-linux/libblkid1_2.38.1-5+b1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libbsd0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ef00f132ddd268ee67d756998723ee18543db27ab34930c0a9f1cff75cf55382",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ef00f132ddd268ee67d756998723ee18543db27ab34930c0a9f1cff75cf55382.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libb/libbsd/libbsd0_0.11.7-2_arm64.deb"],
        deps = ["@debian12_libmd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libbsd0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "bb31cc8b40f962a85b2cec970f7f79cc704a1ae4bad24257a822055404b2c60b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bb31cc8b40f962a85b2cec970f7f79cc704a1ae4bad24257a822055404b2c60b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libb/libbsd/libbsd0_0.11.7-2_amd64.deb"],
        deps = ["@debian12_libmd0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libbz2-1.0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d3a96ece03326498b39ff093a76800dfcbcb1d4049d6ae6e9f6fa1aa7a590ad6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d3a96ece03326498b39ff093a76800dfcbcb1d4049d6ae6e9f6fa1aa7a590ad6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/b/bzip2/libbz2-1.0_1.0.8-5+b1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libbz2-1.0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "54149da3f44b22d523b26b692033b84503d822cc5122fed606ea69cc83ca5aeb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/54149da3f44b22d523b26b692033b84503d822cc5122fed606ea69cc83ca5aeb.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/b/bzip2/libbz2-1.0_1.0.8-5+b1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libc-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "bf333f8ed1ddb406c34a753bc574f9901c2ee41cebfa58d9aff985d8d07c5ba5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bf333f8ed1ddb406c34a753bc574f9901c2ee41cebfa58d9aff985d8d07c5ba5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glibc/libc-bin_2.36-9+deb12u3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libc-bin_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9840ce93b42b66c784852df07ee9131b7acab886177794a5c9ba761da9463887",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9840ce93b42b66c784852df07ee9131b7acab886177794a5c9ba761da9463887.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glibc/libc-bin_2.36-9+deb12u3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libc-dev-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3d0abaae380855474f7631d7f2b6933406216d255ed02f7cf9b1fa9303f1b377",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3d0abaae380855474f7631d7f2b6933406216d255ed02f7cf9b1fa9303f1b377.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glibc/libc-dev-bin_2.36-9+deb12u3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libc-dev-bin_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1a394277e17426a10abdd9293e06fa0f8c31049fe73027608fe9363dda36f25b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1a394277e17426a10abdd9293e06fa0f8c31049fe73027608fe9363dda36f25b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glibc/libc-dev-bin_2.36-9+deb12u3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libc6-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "096b9a880817d3ece3226f2d15aed399b9aa8c3426206515804cb53099d5b881",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/096b9a880817d3ece3226f2d15aed399b9aa8c3426206515804cb53099d5b881.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glibc/libc6-dev_2.36-9+deb12u3_arm64.deb"],
        deps = ["@debian12_libc-dev-bin_aarch64//:all_files", "@debian12_libcrypt-dev_aarch64//:all_files", "@debian12_linux-libc-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libc6-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b10102de6c5f57bd040e9ee2a5fa9a5182a769ecb56a9ac09af4ab5f38131482",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b10102de6c5f57bd040e9ee2a5fa9a5182a769ecb56a9ac09af4ab5f38131482.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glibc/libc6-dev_2.36-9+deb12u3_amd64.deb"],
        deps = ["@debian12_libc-dev-bin_x86_64//:all_files", "@debian12_libcrypt-dev_x86_64//:all_files", "@debian12_linux-libc-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libc6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "75668ed8689fa8c38ce36728f84eb987bb8e7b0acc16802afe2bb07b25c3155c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/75668ed8689fa8c38ce36728f84eb987bb8e7b0acc16802afe2bb07b25c3155c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glibc/libc6_2.36-9+deb12u3_arm64.deb"],
        deps = ["@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libc6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glibc/libc6_2.36-9+deb12u3_amd64.deb"],
        deps = ["@debian12_libgcc-s1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcap-ng0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "24e74ad29a37d2a3940b8977d11298a7afc77379ef414b561d79c64147d740e0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/24e74ad29a37d2a3940b8977d11298a7afc77379ef414b561d79c64147d740e0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libc/libcap-ng/libcap-ng0_0.8.3-1+b3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcap-ng0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b4b54769c77e4a71c8b33aee4d600ba28a9994a1c6f60d55d4ebe7fc44882e07",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b4b54769c77e4a71c8b33aee4d600ba28a9994a1c6f60d55d4ebe7fc44882e07.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libc/libcap-ng/libcap-ng0_0.8.3-1+b3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcap2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c96dbe3a37385c9f0a5d559d55bbd97c3aae649e27ba8f502b78172f78859e46",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c96dbe3a37385c9f0a5d559d55bbd97c3aae649e27ba8f502b78172f78859e46.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libc/libcap2/libcap2_2.66-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcap2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b36fefe9867f9e59b540f952e957a72ebdc241e997179d826da19a9511ade4a3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b36fefe9867f9e59b540f952e957a72ebdc241e997179d826da19a9511ade4a3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libc/libcap2/libcap2_2.66-4_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcom-err2_aarch64",
        sha256 = "0bd07f2a030f9e0346a80bbdf211fe3a55835dda70ed0e9d9da23f0e24cf4e99",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0bd07f2a030f9e0346a80bbdf211fe3a55835dda70ed0e9d9da23f0e24cf4e99.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/e/e2fsprogs/libcom-err2_1.47.0-2_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcom-err2_x86_64",
        sha256 = "8010e4285276bb344c05ae780deae2fffb45e237116c3a78481365c5954125ec",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8010e4285276bb344c05ae780deae2fffb45e237116c3a78481365c5954125ec.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/e/e2fsprogs/libcom-err2_1.47.0-2_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcrypt-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8f40a050f6d54b47b0972f77ea29932573e0e6b4521b833c9d825f46a9db8b75",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8f40a050f6d54b47b0972f77ea29932573e0e6b4521b833c9d825f46a9db8b75.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcrypt/libcrypt-dev_4.4.33-2_arm64.deb"],
        deps = ["@debian12_libcrypt1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcrypt-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "81ccd29130f75a9e3adabc80e61921abff42f76761e1f792fa2d1bb69af7f52f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/81ccd29130f75a9e3adabc80e61921abff42f76761e1f792fa2d1bb69af7f52f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcrypt/libcrypt-dev_4.4.33-2_amd64.deb"],
        deps = ["@debian12_libcrypt1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcrypt1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "eea0ad76ea5eb507127fea0c291622ea4ecdbb71c4b9a8ed9c76ae33fc1a0127",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/eea0ad76ea5eb507127fea0c291622ea4ecdbb71c4b9a8ed9c76ae33fc1a0127.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcrypt/libcrypt1_4.4.33-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libcrypt1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f5f60a5cdfd4e4eaa9438ade5078a57741a7a78d659fcb0c701204f523e8bd29",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f5f60a5cdfd4e4eaa9438ade5078a57741a7a78d659fcb0c701204f523e8bd29.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcrypt/libcrypt1_4.4.33-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdb5.3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "344367608d622298a3d916f4cee3dc3173286f3b21f8f497ab21e7178ba930f9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/344367608d622298a3d916f4cee3dc3173286f3b21f8f497ab21e7178ba930f9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/db5.3/libdb5.3_5.3.28+dfsg2-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdb5.3_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "7dc5127b8dd0da80e992ba594954c005ae4359d839a24eb65d0d8129b5235c84",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7dc5127b8dd0da80e992ba594954c005ae4359d839a24eb65d0d8129b5235c84.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/d/db5.3/libdb5.3_5.3.28+dfsg2-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdevmapper1.02.1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "0b4ffad17467a043dd1cb9cdbbcf8cbaf2ab88e902ae4c110a0b08d74753589b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0b4ffad17467a043dd1cb9cdbbcf8cbaf2ab88e902ae4c110a0b08d74753589b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lvm2/libdevmapper1.02.1_1.02.185-2_arm64.deb"],
        deps = ["@debian12_dmsetup_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files", "@debian12_libudev1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdevmapper1.02.1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "aaa78ca236055fedccf637eacf7bda02bf1980b2db668dccd202b04d0d2cfe04",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/aaa78ca236055fedccf637eacf7bda02bf1980b2db668dccd202b04d0d2cfe04.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lvm2/libdevmapper1.02.1_1.02.185-2_amd64.deb"],
        deps = ["@debian12_dmsetup_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files", "@debian12_libudev1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-amdgpu1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "24692e933b98b2b8b80d87c45728efa2bc7176120ce9efcb8793407e508b5a32",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/24692e933b98b2b8b80d87c45728efa2bc7176120ce9efcb8793407e508b5a32.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-amdgpu1_2.4.114-1+b1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-amdgpu1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b75a71e96f1faac0f131ac657e09efcbe8968eef62cc34b8abfcff2ff9f0cccd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b75a71e96f1faac0f131ac657e09efcbe8968eef62cc34b8abfcff2ff9f0cccd.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-amdgpu1_2.4.114-1+b1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "32f9664138b38b224383c6986457d5ad2ec8efd559b1a0ce7749405f7a451aad",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/32f9664138b38b224383c6986457d5ad2ec8efd559b1a0ce7749405f7a451aad.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-common_2.4.114-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-common_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "32f9664138b38b224383c6986457d5ad2ec8efd559b1a0ce7749405f7a451aad",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/32f9664138b38b224383c6986457d5ad2ec8efd559b1a0ce7749405f7a451aad.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-common_2.4.114-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "26d7d0dcbbda8ed023b57989e0fca7f0ff8372daaf654daf8b3a362557be5e21",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/26d7d0dcbbda8ed023b57989e0fca7f0ff8372daaf654daf8b3a362557be5e21.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-dev_2.4.114-1+b1_arm64.deb"],
        deps = ["@debian12_libdrm-amdgpu1_aarch64//:all_files", "@debian12_libdrm-etnaviv1_aarch64//:all_files", "@debian12_libdrm-freedreno1_aarch64//:all_files", "@debian12_libdrm-nouveau2_aarch64//:all_files", "@debian12_libdrm-radeon1_aarch64//:all_files", "@debian12_libdrm-tegra0_aarch64//:all_files", "@debian12_libdrm2_aarch64//:all_files", "@debian12_libpciaccess-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "073e7ac28672c7fb55141cf39ba1daa25667bd578c76dc43eb4f911b8d3a259a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/073e7ac28672c7fb55141cf39ba1daa25667bd578c76dc43eb4f911b8d3a259a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-dev_2.4.114-1+b1_amd64.deb"],
        deps = ["@debian12_libdrm-amdgpu1_x86_64//:all_files", "@debian12_libdrm-intel1_x86_64//:all_files", "@debian12_libdrm-nouveau2_x86_64//:all_files", "@debian12_libdrm-radeon1_x86_64//:all_files", "@debian12_libdrm2_x86_64//:all_files", "@debian12_libpciaccess-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-etnaviv1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "be1e16ac0ce067dc4e8b5b6cb1fdafc838eb66bf6e13dbdb0a791df14d8ba4ac",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/be1e16ac0ce067dc4e8b5b6cb1fdafc838eb66bf6e13dbdb0a791df14d8ba4ac.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-etnaviv1_2.4.114-1+b1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-freedreno1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d2e90c205bd4e49ead398d8dbdf9232d651aa16961ad2ded9bc14e0d3e6796b0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d2e90c205bd4e49ead398d8dbdf9232d651aa16961ad2ded9bc14e0d3e6796b0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-freedreno1_2.4.114-1+b1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-intel1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b0e39318d14c07f4d85668b6da7f66a1341addf87a47d785c34d5a8b393f544c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b0e39318d14c07f4d85668b6da7f66a1341addf87a47d785c34d5a8b393f544c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-intel1_2.4.114-1+b1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files", "@debian12_libpciaccess0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-nouveau2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2acf0db609e8569dd89eea5f3e6f27968de50474d31213665901ad1c6d1133de",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2acf0db609e8569dd89eea5f3e6f27968de50474d31213665901ad1c6d1133de.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-nouveau2_2.4.114-1+b1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-nouveau2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ba59bb9ec6e1baf59fc4d4eb095a524e40d045af2413dad9d28df517005388b6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ba59bb9ec6e1baf59fc4d4eb095a524e40d045af2413dad9d28df517005388b6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-nouveau2_2.4.114-1+b1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-radeon1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e53c2b4be53b4a8e6413887525c9172bdc95ed3a140a52ce23d6e1a68789aaa0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e53c2b4be53b4a8e6413887525c9172bdc95ed3a140a52ce23d6e1a68789aaa0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-radeon1_2.4.114-1+b1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-radeon1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2da3a9233187f995ad5a3e6db3d37252ea7209f0ca9605484d03478ebcc15feb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2da3a9233187f995ad5a3e6db3d37252ea7209f0ca9605484d03478ebcc15feb.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-radeon1_2.4.114-1+b1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm-tegra0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c47ae757624a1740e5160683a31751c77725d8c93f2e626308a9f0380000f0d3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c47ae757624a1740e5160683a31751c77725d8c93f2e626308a9f0380000f0d3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm-tegra0_2.4.114-1+b1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f5f15a46d02cf5d9fa52d4f1c54b8cf80c398711ad771a9938b12399b8d8090c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f5f15a46d02cf5d9fa52d4f1c54b8cf80c398711ad771a9938b12399b8d8090c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm2_2.4.114-1+b1_arm64.deb"],
        deps = ["@debian12_libdrm-common_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libdrm2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "be18fb670797ba32da9628cf3e8acd83160d8db8c8dd842501dd8e401c3b5371",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/be18fb670797ba32da9628cf3e8acd83160d8db8c8dd842501dd8e401c3b5371.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libd/libdrm/libdrm2_2.4.114-1+b1_amd64.deb"],
        deps = ["@debian12_libdrm-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libedit2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6ca1e355586ebfcfe8122b4d317fcaad4a9a10ca455c6a9bbd36f55bf4be85d3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6ca1e355586ebfcfe8122b4d317fcaad4a9a10ca455c6a9bbd36f55bf4be85d3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libe/libedit/libedit2_3.1-20221030-2_arm64.deb"],
        deps = ["@debian12_libbsd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libedit2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1cf14abf2716d3279db12d0657a5737cf70074a1e71d3bdf73206625e3c89ce6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1cf14abf2716d3279db12d0657a5737cf70074a1e71d3bdf73206625e3c89ce6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libe/libedit/libedit2_3.1-20221030-2_amd64.deb"],
        deps = ["@debian12_libbsd0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8a3004f892785a5fa4c9f969c116f567a36d1c8e9b3d5fd0403055151e36c7d3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8a3004f892785a5fa4c9f969c116f567a36d1c8e9b3d5fd0403055151e36c7d3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libegl-dev_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libegl1_aarch64//:all_files", "@debian12_libgl-dev_aarch64//:all_files", "@debian12_libx11-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "0ec9bffb024527dfb61eaeae44966e2b5b3114b3dc315b4117b0df95b1c96ef8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0ec9bffb024527dfb61eaeae44966e2b5b3114b3dc315b4117b0df95b1c96ef8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libegl-dev_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libegl1_x86_64//:all_files", "@debian12_libgl-dev_x86_64//:all_files", "@debian12_libx11-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl-mesa0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e00e8122e2c68742f3306b90f4be93508c12c8714350818c1a4043e7e314e0d1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e00e8122e2c68742f3306b90f4be93508c12c8714350818c1a4043e7e314e0d1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libegl-mesa0_22.3.6-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files", "@debian12_libexpat1_aarch64//:all_files", "@debian12_libgbm1_aarch64//:all_files", "@debian12_libglapi-mesa_aarch64//:all_files", "@debian12_libwayland-client0_aarch64//:all_files", "@debian12_libwayland-server0_aarch64//:all_files", "@debian12_libx11-xcb1_aarch64//:all_files", "@debian12_libxcb-dri2-0_aarch64//:all_files", "@debian12_libxcb-dri3-0_aarch64//:all_files", "@debian12_libxcb-present0_aarch64//:all_files", "@debian12_libxcb-randr0_aarch64//:all_files", "@debian12_libxcb-sync1_aarch64//:all_files", "@debian12_libxcb-xfixes0_aarch64//:all_files", "@debian12_libxcb1_aarch64//:all_files", "@debian12_libxshmfence1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl-mesa0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f3d97aa976fbebcb9bd8ddb6a0fdd8406b5c6be350d9031802b0e358aead197b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f3d97aa976fbebcb9bd8ddb6a0fdd8406b5c6be350d9031802b0e358aead197b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libegl-mesa0_22.3.6-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files", "@debian12_libexpat1_x86_64//:all_files", "@debian12_libgbm1_x86_64//:all_files", "@debian12_libgcc-s1_x86_64//:all_files", "@debian12_libglapi-mesa_x86_64//:all_files", "@debian12_libwayland-client0_x86_64//:all_files", "@debian12_libwayland-server0_x86_64//:all_files", "@debian12_libx11-xcb1_x86_64//:all_files", "@debian12_libxcb-dri2-0_x86_64//:all_files", "@debian12_libxcb-dri3-0_x86_64//:all_files", "@debian12_libxcb-present0_x86_64//:all_files", "@debian12_libxcb-randr0_x86_64//:all_files", "@debian12_libxcb-sync1_x86_64//:all_files", "@debian12_libxcb-xfixes0_x86_64//:all_files", "@debian12_libxcb1_x86_64//:all_files", "@debian12_libxshmfence1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl1-mesa-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "730e226fc1a0053a8e4a49e428ff3752124e039d33a42e11e4363d19db60b667",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/730e226fc1a0053a8e4a49e428ff3752124e039d33a42e11e4363d19db60b667.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libegl1-mesa-dev_22.3.6-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libegl-dev_aarch64//:all_files", "@debian12_libglvnd-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl1-mesa-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2d85f95feed901fc02dc3476d7bd1752a82b15e42e63805eedad574718c9350f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2d85f95feed901fc02dc3476d7bd1752a82b15e42e63805eedad574718c9350f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libegl1-mesa-dev_22.3.6-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libegl-dev_x86_64//:all_files", "@debian12_libglvnd-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl1-mesa_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4ef35b5429236c29103fdd804f3a936787e9a16e75c8c615c765f4cce0c5bce9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4ef35b5429236c29103fdd804f3a936787e9a16e75c8c615c765f4cce0c5bce9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libegl1-mesa_22.3.6-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libegl1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl1-mesa_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ea38c6863faf8696f69a40907c85ac719ea49d586f156c48a54f1b2cc8b93317",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ea38c6863faf8696f69a40907c85ac719ea49d586f156c48a54f1b2cc8b93317.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libegl1-mesa_22.3.6-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libegl1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "707097a275155c600e2e9251c4ee7cdfdb2d8f50a678a850a2526a7bc9664166",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/707097a275155c600e2e9251c4ee7cdfdb2d8f50a678a850a2526a7bc9664166.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libegl1_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libegl-mesa0_aarch64//:all_files", "@debian12_libglvnd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libegl1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fe4d8b39f6e6fe1a32ab1efd85893553eaa9cf3866aa668ccf355f585b37d523",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fe4d8b39f6e6fe1a32ab1efd85893553eaa9cf3866aa668ccf355f585b37d523.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libegl1_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libegl-mesa0_x86_64//:all_files", "@debian12_libglvnd0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libelf-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fe16ede10e1b80a2f882f0e83c52695e77280c61eebe7106f016d4025e82399b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fe16ede10e1b80a2f882f0e83c52695e77280c61eebe7106f016d4025e82399b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/e/elfutils/libelf-dev_0.188-2.1_arm64.deb"],
        deps = ["@debian12_zlib1g-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libelf-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "79cb66b55021bd0130308369524bac5240d0b5463cb252cd44be6a1500fdebec",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/79cb66b55021bd0130308369524bac5240d0b5463cb252cd44be6a1500fdebec.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/e/elfutils/libelf-dev_0.188-2.1_amd64.deb"],
        deps = ["@debian12_zlib1g-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libelf1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "32971952d8f5d563447bf275f24e26057500924da2d855c1edb53b0f0400bd11",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/32971952d8f5d563447bf275f24e26057500924da2d855c1edb53b0f0400bd11.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/e/elfutils/libelf1_0.188-2.1_arm64.deb"],
        deps = ["@debian12_zlib1g_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libelf1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "619add379c606b3ac6c1a175853b918e6939598a83d8ebadf3bdfd50d10b3c8c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/619add379c606b3ac6c1a175853b918e6939598a83d8ebadf3bdfd50d10b3c8c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/e/elfutils/libelf1_0.188-2.1_amd64.deb"],
        deps = ["@debian12_zlib1g_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libexpat1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "39de7d17cb312d76f586866a38d7649102178a2cdb7f4cef1b4f279ea3cebf07",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/39de7d17cb312d76f586866a38d7649102178a2cdb7f4cef1b4f279ea3cebf07.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/e/expat/libexpat1_2.5.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libexpat1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fe36a7f35361fc40d0057ef447a7302fd41d51740d51c98fb3870bbed5b96e56",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fe36a7f35361fc40d0057ef447a7302fd41d51740d51c98fb3870bbed5b96e56.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/e/expat/libexpat1_2.5.0-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libffi-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8ff6335f76752ef8659762435d0d88129ef0090a587520853238809bd10ef392",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8ff6335f76752ef8659762435d0d88129ef0090a587520853238809bd10ef392.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libf/libffi/libffi-dev_3.4.4-1_arm64.deb"],
        deps = ["@debian12_libffi8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libffi-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "89fb890aee5148f4d308a46cd8980a54fd44135f068f05b38a6ad06800bf6df3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/89fb890aee5148f4d308a46cd8980a54fd44135f068f05b38a6ad06800bf6df3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libf/libffi/libffi-dev_3.4.4-1_amd64.deb"],
        deps = ["@debian12_libffi8_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libffi8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "80b5c36177dc0e29d531c7eddbed3cc7355cb490e49f8cfa5959572d161f27b3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/80b5c36177dc0e29d531c7eddbed3cc7355cb490e49f8cfa5959572d161f27b3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libf/libffi/libffi8_3.4.4-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libffi8_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6d9f6c25c30efccce6d4bceaa48ea86c329a3432abb360a141f76ac223a4c34a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6d9f6c25c30efccce6d4bceaa48ea86c329a3432abb360a141f76ac223a4c34a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libf/libffi/libffi8_3.4.4-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libfile-find-rule-perl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2d7e1ab5eddf439954881d9f53d8c99ff7508754ce49de5250fab79808a85f63",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2d7e1ab5eddf439954881d9f53d8c99ff7508754ce49de5250fab79808a85f63.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libf/libfile-find-rule-perl/libfile-find-rule-perl_0.34-3_all.deb"],
        deps = ["@debian12_libnumber-compare-perl_aarch64//:all_files", "@debian12_libtext-glob-perl_aarch64//:all_files", "@debian12_perl_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libfile-find-rule-perl_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2d7e1ab5eddf439954881d9f53d8c99ff7508754ce49de5250fab79808a85f63",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2d7e1ab5eddf439954881d9f53d8c99ff7508754ce49de5250fab79808a85f63.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libf/libfile-find-rule-perl/libfile-find-rule-perl_0.34-3_all.deb"],
        deps = ["@debian12_libnumber-compare-perl_x86_64//:all_files", "@debian12_libtext-glob-perl_x86_64//:all_files", "@debian12_perl_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgbm1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c76790e11aac46e328b6a13b34cccd3ef01fad79cfe80d2c4aa2384ac9ddb1f8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c76790e11aac46e328b6a13b34cccd3ef01fad79cfe80d2c4aa2384ac9ddb1f8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libgbm1_22.3.6-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files", "@debian12_libexpat1_aarch64//:all_files", "@debian12_libwayland-server0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgbm1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b0edffc231b3261eedbed7d1fafabf1f1cc04ca3c149c2be8322ec70dd17e786",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b0edffc231b3261eedbed7d1fafabf1f1cc04ca3c149c2be8322ec70dd17e786.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libgbm1_22.3.6-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files", "@debian12_libexpat1_x86_64//:all_files", "@debian12_libwayland-server0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgcc-12-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fb74956b50ec86fd82a7a20aede3047e728155da019c34ff4fcbd3e3b79bbda8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fb74956b50ec86fd82a7a20aede3047e728155da019c34ff4fcbd3e3b79bbda8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libgcc-12-dev_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files", "@debian12_libasan8_aarch64//:all_files", "@debian12_libatomic1_aarch64//:all_files", "@debian12_libgcc-s1_aarch64//:all_files", "@debian12_libgomp1_aarch64//:all_files", "@debian12_libhwasan0_aarch64//:all_files", "@debian12_libitm1_aarch64//:all_files", "@debian12_liblsan0_aarch64//:all_files", "@debian12_libtsan2_aarch64//:all_files", "@debian12_libubsan1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgcc-12-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6ffd3721915c49580fc9bcf1ef06deab4ad59e99c52c9f349d03954642b97655",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6ffd3721915c49580fc9bcf1ef06deab4ad59e99c52c9f349d03954642b97655.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libgcc-12-dev_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files", "@debian12_libasan8_x86_64//:all_files", "@debian12_libatomic1_x86_64//:all_files", "@debian12_libgcc-s1_x86_64//:all_files", "@debian12_libgomp1_x86_64//:all_files", "@debian12_libitm1_x86_64//:all_files", "@debian12_liblsan0_x86_64//:all_files", "@debian12_libquadmath0_x86_64//:all_files", "@debian12_libtsan2_x86_64//:all_files", "@debian12_libubsan1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgcc-s1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6fce2268d8f3152a4e84634f5a24133d3c62903b2f9b11b9c59235cbbc1b23a8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6fce2268d8f3152a4e84634f5a24133d3c62903b2f9b11b9c59235cbbc1b23a8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libgcc-s1_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgcc-s1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f3d1d48c0599aea85b7f2077a01d285badc42998c1a1e7473935d5cf995c8141",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f3d1d48c0599aea85b7f2077a01d285badc42998c1a1e7473935d5cf995c8141.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libgcc-s1_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgcrypt20_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "79fc67c21684689728c8320d8a2b0a7204df21dc4c0da4fae3828ceb389e2ba2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/79fc67c21684689728c8320d8a2b0a7204df21dc4c0da4fae3828ceb389e2ba2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libgcrypt20/libgcrypt20_1.10.1-3_arm64.deb"],
        deps = ["@debian12_libgpg-error0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgcrypt20_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "bffcac7e4f69e39d37d4a33e841d6371ac8b5aba6cd55546b385dc7ff6c702f5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bffcac7e4f69e39d37d4a33e841d6371ac8b5aba6cd55546b385dc7ff6c702f5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libgcrypt20/libgcrypt20_1.10.1-3_amd64.deb"],
        deps = ["@debian12_libgpg-error0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgdbm-compat4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "acc13a12acbf3b17dbacc77e3c22c6c273a552150cd50fdc9bd70cdb5169af73",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/acc13a12acbf3b17dbacc77e3c22c6c273a552150cd50fdc9bd70cdb5169af73.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gdbm/libgdbm-compat4_1.23-3_arm64.deb"],
        deps = ["@debian12_libgdbm6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgdbm-compat4_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4af36a590b68d415a78d9238b932b6a4579f515ec8a8016597498acff5b515a4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4af36a590b68d415a78d9238b932b6a4579f515ec8a8016597498acff5b515a4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gdbm/libgdbm-compat4_1.23-3_amd64.deb"],
        deps = ["@debian12_libgdbm6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgdbm6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3c704feaf89c5f2709a76b84126ac8affe4ddff69dc0b91af0ca9910c7e75714",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3c704feaf89c5f2709a76b84126ac8affe4ddff69dc0b91af0ca9910c7e75714.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gdbm/libgdbm6_1.23-3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgdbm6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "95fe4a1336532450e67bd067892f46eaa484139919ea8d067a9ffcbf5a4bf883",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/95fe4a1336532450e67bd067892f46eaa484139919ea8d067a9ffcbf5a4bf883.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gdbm/libgdbm6_1.23-3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgl-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "56add9e710c072cf56504beed50a639f9f7c64f4e31e66958fa65a566084888c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/56add9e710c072cf56504beed50a639f9f7c64f4e31e66958fa65a566084888c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgl-dev_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libgl1_aarch64//:all_files", "@debian12_libglx-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgl-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ebc12df48ae53924e114d9358ef3da4306d7ef8f7179300af52f1faef8b5db3e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ebc12df48ae53924e114d9358ef3da4306d7ef8f7179300af52f1faef8b5db3e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgl-dev_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libgl1_x86_64//:all_files", "@debian12_libglx-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgl1-mesa-dri_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "44f6b12346dc862d1e2031d9b8eab50823d86b89aa80570ba1e8a38a7702b174",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/44f6b12346dc862d1e2031d9b8eab50823d86b89aa80570ba1e8a38a7702b174.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libgl1-mesa-dri_22.3.6-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libdrm-amdgpu1_aarch64//:all_files", "@debian12_libdrm-nouveau2_aarch64//:all_files", "@debian12_libdrm-radeon1_aarch64//:all_files", "@debian12_libdrm2_aarch64//:all_files", "@debian12_libexpat1_aarch64//:all_files", "@debian12_libgcc-s1_aarch64//:all_files", "@debian12_libllvm15_aarch64//:all_files", "@debian12_libsensors5_aarch64//:all_files", "@debian12_libxcb-dri3-0_aarch64//:all_files", "@debian12_libzstd1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgl1-mesa-dri_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2c3fbfb89a8eadf6a82836b2443bc4d9578cb83847daf37cf6ab6074daf4fcda",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2c3fbfb89a8eadf6a82836b2443bc4d9578cb83847daf37cf6ab6074daf4fcda.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libgl1-mesa-dri_22.3.6-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libdrm-amdgpu1_x86_64//:all_files", "@debian12_libdrm-intel1_x86_64//:all_files", "@debian12_libdrm-nouveau2_x86_64//:all_files", "@debian12_libdrm-radeon1_x86_64//:all_files", "@debian12_libdrm2_x86_64//:all_files", "@debian12_libexpat1_x86_64//:all_files", "@debian12_libgcc-s1_x86_64//:all_files", "@debian12_libllvm15_x86_64//:all_files", "@debian12_libsensors5_x86_64//:all_files", "@debian12_libxcb-dri3-0_x86_64//:all_files", "@debian12_libzstd1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgl1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f4043ab47f52325973d2b608514b3469b055b65c744861c7f378755a32e799d6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f4043ab47f52325973d2b608514b3469b055b65c744861c7f378755a32e799d6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgl1_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libglvnd0_aarch64//:all_files", "@debian12_libglx0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgl1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6f89b1702c48e9a2437bb3c1ffac8e1ab2d828fc28b3d14b2eecd4cc19b2c790",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6f89b1702c48e9a2437bb3c1ffac8e1ab2d828fc28b3d14b2eecd4cc19b2c790.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgl1_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libglvnd0_x86_64//:all_files", "@debian12_libglx0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglapi-mesa_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "0714893b42f558e229d6c1c0f83057d49d64b85790bc22bb76566fbcfe62fde9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0714893b42f558e229d6c1c0f83057d49d64b85790bc22bb76566fbcfe62fde9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libglapi-mesa_22.3.6-1+deb12u1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglapi-mesa_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fda45e2e2980cc8fd8e12e401460a702e6f990952549adda5608c2f901c3199c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fda45e2e2980cc8fd8e12e401460a702e6f990952549adda5608c2f901c3199c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libglapi-mesa_22.3.6-1+deb12u1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "941e9963d64479544691f7183fe0eeae32b53b633df71c4c51263517ab2ea8f3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/941e9963d64479544691f7183fe0eeae32b53b633df71c4c51263517ab2ea8f3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgles-dev_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libegl-dev_aarch64//:all_files", "@debian12_libgl-dev_aarch64//:all_files", "@debian12_libgles1_aarch64//:all_files", "@debian12_libgles2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "901b06c4a5223560617e9362d26ab349f9d9dd3d79e7d87d6ce0b06a7c7128f4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/901b06c4a5223560617e9362d26ab349f9d9dd3d79e7d87d6ce0b06a7c7128f4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgles-dev_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libegl-dev_x86_64//:all_files", "@debian12_libgl-dev_x86_64//:all_files", "@debian12_libgles1_x86_64//:all_files", "@debian12_libgles2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "672256ffce24a7a5d4a5b513eb63e04baaa3df779c08513c46c37f66e2e4c1dd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/672256ffce24a7a5d4a5b513eb63e04baaa3df779c08513c46c37f66e2e4c1dd.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgles1_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libglvnd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "aa9a2ef3b275d01a95d7ed56d99a752a0e5d1c1dc828cdfd088ffa15a09750fc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/aa9a2ef3b275d01a95d7ed56d99a752a0e5d1c1dc828cdfd088ffa15a09750fc.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgles1_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libglvnd0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles2-mesa-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c1697c38698bf5b629066cdc4dca3ff40ce59e1e9cf30db3d7e5f7c0ca6a711a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c1697c38698bf5b629066cdc4dca3ff40ce59e1e9cf30db3d7e5f7c0ca6a711a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libgles2-mesa-dev_22.3.6-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libgles-dev_aarch64//:all_files", "@debian12_libglvnd-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles2-mesa-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "0d1f3701ebca2d0db09658d506aba56f86fd144f0a94900d33749e8816428a54",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0d1f3701ebca2d0db09658d506aba56f86fd144f0a94900d33749e8816428a54.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libgles2-mesa-dev_22.3.6-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libgles-dev_x86_64//:all_files", "@debian12_libglvnd-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles2-mesa_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3216e18d5c1199225f5144d3a279ff7b1f6fbd33a022fa2f515d1d390996f24f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3216e18d5c1199225f5144d3a279ff7b1f6fbd33a022fa2f515d1d390996f24f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libgles2-mesa_22.3.6-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libgles2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles2-mesa_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "98f33e56590f262d81157d9b43f006fb60f8929b11327d38177001bd041e2f37",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/98f33e56590f262d81157d9b43f006fb60f8929b11327d38177001bd041e2f37.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libgles2-mesa_22.3.6-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libgles2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "efeb2717380f8411d66b3f669bb99bbd83ff09db3d4a1661533aa31af659608c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/efeb2717380f8411d66b3f669bb99bbd83ff09db3d4a1661533aa31af659608c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgles2_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libglvnd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgles2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "07b2f51b8aa3c8d6d928133cd46087bd8793d0d67c203f09fc289d45a2cf5f47",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/07b2f51b8aa3c8d6d928133cd46087bd8793d0d67c203f09fc289d45a2cf5f47.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libgles2_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libglvnd0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglib2.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "203b248ce95efd67b89586849a027e8691aaf3ad1df5fb263fff949fa4ba2af4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/203b248ce95efd67b89586849a027e8691aaf3ad1df5fb263fff949fa4ba2af4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glib2.0/libglib2.0-0_2.74.6-2_arm64.deb"],
        deps = ["@debian12_libffi8_aarch64//:all_files", "@debian12_libmount1_aarch64//:all_files", "@debian12_libpcre2-8-0_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglib2.0-0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "7e90ba4670377ae29f1a718798b4d5e088ac97d2dfa20a13b4b2ea357b61ec29",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7e90ba4670377ae29f1a718798b4d5e088ac97d2dfa20a13b4b2ea357b61ec29.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/glib2.0/libglib2.0-0_2.74.6-2_amd64.deb"],
        deps = ["@debian12_libffi8_x86_64//:all_files", "@debian12_libmount1_x86_64//:all_files", "@debian12_libpcre2-8-0_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglvnd-core-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a27c9b7490f9051b6128ba4e2b7ab394cda162946f38eba080520cae41af7ac2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a27c9b7490f9051b6128ba4e2b7ab394cda162946f38eba080520cae41af7ac2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglvnd-core-dev_1.6.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglvnd-core-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "57f3e4f1d92c270ba9b8ab13ef82689a6b080d135b1deec546d6fa9095cc43f7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/57f3e4f1d92c270ba9b8ab13ef82689a6b080d135b1deec546d6fa9095cc43f7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglvnd-core-dev_1.6.0-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglvnd-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "86cba4e243d10ae07cc6e769f23ecd097b2f87b722dc5ac48a48a3794cd29513",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/86cba4e243d10ae07cc6e769f23ecd097b2f87b722dc5ac48a48a3794cd29513.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglvnd-dev_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libegl-dev_aarch64//:all_files", "@debian12_libgl-dev_aarch64//:all_files", "@debian12_libglvnd-core-dev_aarch64//:all_files", "@debian12_libglvnd0_aarch64//:all_files", "@debian12_libglx-dev_aarch64//:all_files", "@debian12_libopengl-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglvnd-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "198f1ab72ead7ff4963875ebc239183f33f0e8171074204a59a6c493f208bfec",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/198f1ab72ead7ff4963875ebc239183f33f0e8171074204a59a6c493f208bfec.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglvnd-dev_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libegl-dev_x86_64//:all_files", "@debian12_libgl-dev_x86_64//:all_files", "@debian12_libglvnd-core-dev_x86_64//:all_files", "@debian12_libglvnd0_x86_64//:all_files", "@debian12_libglx-dev_x86_64//:all_files", "@debian12_libopengl-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglvnd0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d5063205fab5322ee686f33f87c6315b3fcd1335ea4e7bc8202b48ed4f9c9e83",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d5063205fab5322ee686f33f87c6315b3fcd1335ea4e7bc8202b48ed4f9c9e83.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglvnd0_1.6.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglvnd0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b6da5b153dd62d8b5e5fbe25242db1fc05c068707c365db49abda8c2427c75f8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b6da5b153dd62d8b5e5fbe25242db1fc05c068707c365db49abda8c2427c75f8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglvnd0_1.6.0-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglx-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f3d6fca16c259743208df6f845aee3189eb54fed00e159811135089c16d0ad9c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f3d6fca16c259743208df6f845aee3189eb54fed00e159811135089c16d0ad9c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglx-dev_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libglx0_aarch64//:all_files", "@debian12_libx11-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglx-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "be1aa9b3a316aff17b912d473a8b2cba62620c9fcaa0102945f641981c3e2d5e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/be1aa9b3a316aff17b912d473a8b2cba62620c9fcaa0102945f641981c3e2d5e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglx-dev_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libglx0_x86_64//:all_files", "@debian12_libx11-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglx-mesa0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e9b9d61013955378228468a2489e32dcf736f89e95cb815189e65e16eeac5d90",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e9b9d61013955378228468a2489e32dcf736f89e95cb815189e65e16eeac5d90.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libglx-mesa0_22.3.6-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files", "@debian12_libexpat1_aarch64//:all_files", "@debian12_libgl1-mesa-dri_aarch64//:all_files", "@debian12_libglapi-mesa_aarch64//:all_files", "@debian12_libx11-6_aarch64//:all_files", "@debian12_libx11-xcb1_aarch64//:all_files", "@debian12_libxcb-dri2-0_aarch64//:all_files", "@debian12_libxcb-dri3-0_aarch64//:all_files", "@debian12_libxcb-glx0_aarch64//:all_files", "@debian12_libxcb-present0_aarch64//:all_files", "@debian12_libxcb-randr0_aarch64//:all_files", "@debian12_libxcb-shm0_aarch64//:all_files", "@debian12_libxcb-sync1_aarch64//:all_files", "@debian12_libxcb-xfixes0_aarch64//:all_files", "@debian12_libxcb1_aarch64//:all_files", "@debian12_libxext6_aarch64//:all_files", "@debian12_libxfixes3_aarch64//:all_files", "@debian12_libxshmfence1_aarch64//:all_files", "@debian12_libxxf86vm1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglx-mesa0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6d0da356b1aaf73f85233773bb41506cc84e4744aedc69ef1db92848956de621",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6d0da356b1aaf73f85233773bb41506cc84e4744aedc69ef1db92848956de621.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/libglx-mesa0_22.3.6-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files", "@debian12_libexpat1_x86_64//:all_files", "@debian12_libgl1-mesa-dri_x86_64//:all_files", "@debian12_libglapi-mesa_x86_64//:all_files", "@debian12_libx11-6_x86_64//:all_files", "@debian12_libx11-xcb1_x86_64//:all_files", "@debian12_libxcb-dri2-0_x86_64//:all_files", "@debian12_libxcb-dri3-0_x86_64//:all_files", "@debian12_libxcb-glx0_x86_64//:all_files", "@debian12_libxcb-present0_x86_64//:all_files", "@debian12_libxcb-randr0_x86_64//:all_files", "@debian12_libxcb-shm0_x86_64//:all_files", "@debian12_libxcb-sync1_x86_64//:all_files", "@debian12_libxcb-xfixes0_x86_64//:all_files", "@debian12_libxcb1_x86_64//:all_files", "@debian12_libxext6_x86_64//:all_files", "@debian12_libxfixes3_x86_64//:all_files", "@debian12_libxshmfence1_x86_64//:all_files", "@debian12_libxxf86vm1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglx0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "523f9b71f75e48ae605d8ef298a4b03ade26ade9a808a85329a3bd82a54b9b0d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/523f9b71f75e48ae605d8ef298a4b03ade26ade9a808a85329a3bd82a54b9b0d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglx0_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libglx-mesa0_aarch64//:all_files", "@debian12_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libglx0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "95f568df73dedf43ae66834a75502112e0d4f3ad7124f3dbfa790b739383b896",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/95f568df73dedf43ae66834a75502112e0d4f3ad7124f3dbfa790b739383b896.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libglx0_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libglx-mesa0_x86_64//:all_files", "@debian12_libx11-6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgmp10_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9906387c1dd806518c915bd8616d072c741061d7fa26b222e52763456060b31a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9906387c1dd806518c915bd8616d072c741061d7fa26b222e52763456060b31a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gmp/libgmp10_6.2.1+dfsg1-1.1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgmp10_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "187aedef2ed763f425c1e523753b9719677633c7eede660401739e9c893482bd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/187aedef2ed763f425c1e523753b9719677633c7eede660401739e9c893482bd.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gmp/libgmp10_6.2.1+dfsg1-1.1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgnutls30_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8b0cffd9812e93b731b11085b8f30f2fe64a7cca439ab26b02cbbdc1ecf1c722",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8b0cffd9812e93b731b11085b8f30f2fe64a7cca439ab26b02cbbdc1ecf1c722.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnutls28/libgnutls30_3.7.9-2+deb12u1_arm64.deb"],
        deps = ["@debian12_libgmp10_aarch64//:all_files", "@debian12_libhogweed6_aarch64//:all_files", "@debian12_libidn2-0_aarch64//:all_files", "@debian12_libnettle8_aarch64//:all_files", "@debian12_libp11-kit0_aarch64//:all_files", "@debian12_libtasn1-6_aarch64//:all_files", "@debian12_libunistring2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgnutls30_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b9c15ab69bb1408136f094e593bb9bedc1dec4a830519c412a191e4ca6d1a287",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b9c15ab69bb1408136f094e593bb9bedc1dec4a830519c412a191e4ca6d1a287.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gnutls28/libgnutls30_3.7.9-2+deb12u1_amd64.deb"],
        deps = ["@debian12_libgmp10_x86_64//:all_files", "@debian12_libhogweed6_x86_64//:all_files", "@debian12_libidn2-0_x86_64//:all_files", "@debian12_libnettle8_x86_64//:all_files", "@debian12_libp11-kit0_x86_64//:all_files", "@debian12_libtasn1-6_x86_64//:all_files", "@debian12_libunistring2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgomp1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a2fd2803bf03384ac90a54f1179a29f2fb3c192f3ff483a3dd8ec6c3351ce5d0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a2fd2803bf03384ac90a54f1179a29f2fb3c192f3ff483a3dd8ec6c3351ce5d0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libgomp1_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgomp1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1dbc499d2055cb128fa4ed678a7adbcced3d882b3509e26d5aa3742a4b9e5b2f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1dbc499d2055cb128fa4ed678a7adbcced3d882b3509e26d5aa3742a4b9e5b2f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libgomp1_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgpg-error0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "aff6ce011ae9abf7090e906f0cf6bc2b447bbc4cc7e03ff117f9d73528857352",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/aff6ce011ae9abf7090e906f0cf6bc2b447bbc4cc7e03ff117f9d73528857352.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libgpg-error/libgpg-error0_1.46-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgpg-error0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "89944ee11d7370ce6ef46fc52f094c4a6512eff8943ec4c6ebefeae6360ceada",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/89944ee11d7370ce6ef46fc52f094c4a6512eff8943ec4c6ebefeae6360ceada.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libgpg-error/libgpg-error0_1.46-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgpgme11_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "adf21a9ee8c3005804afcbaccdb3fe73023fd18e91babd55100b0c2a2813a578",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/adf21a9ee8c3005804afcbaccdb3fe73023fd18e91babd55100b0c2a2813a578.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gpgme1.0/libgpgme11_1.18.0-3+b1_arm64.deb"],
        deps = ["@debian12_gnupg_aarch64//:all_files", "@debian12_libassuan0_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgpgme11_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "dc075584050dc5c8ac27563fc222e8c1ea71128a019a6d129d5823e47ac1e55e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/dc075584050dc5c8ac27563fc222e8c1ea71128a019a6d129d5823e47ac1e55e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gpgme1.0/libgpgme11_1.18.0-3+b1_amd64.deb"],
        deps = ["@debian12_gnupg_x86_64//:all_files", "@debian12_libassuan0_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgssapi-krb5-2_aarch64",
        sha256 = "94bf0e0fd3c51bd3b082cd360facf1e1db39d825fe93b581d673283edb64c1d4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/94bf0e0fd3c51bd3b082cd360facf1e1db39d825fe93b581d673283edb64c1d4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/krb5/libgssapi-krb5-2_1.20.1-2+deb12u1_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libcom-err2_aarch64//:all_files", "@debian12_libk5crypto3_aarch64//:all_files", "@debian12_libkrb5-3_aarch64//:all_files", "@debian12_libkrb5support0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libgssapi-krb5-2_x86_64",
        sha256 = "6631304ce4b5b9ba0af3fdebf088a734aed2d28ffad2a03ba79e4fcb2e226dd6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6631304ce4b5b9ba0af3fdebf088a734aed2d28ffad2a03ba79e4fcb2e226dd6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/krb5/libgssapi-krb5-2_1.20.1-2+deb12u1_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libcom-err2_x86_64//:all_files", "@debian12_libk5crypto3_x86_64//:all_files", "@debian12_libkrb5-3_x86_64//:all_files", "@debian12_libkrb5support0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libhogweed6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e653a1a7e5a44be0f7b6443dc6ac865d2504e49149660fc253655245965e157f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e653a1a7e5a44be0f7b6443dc6ac865d2504e49149660fc253655245965e157f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/nettle/libhogweed6_3.8.1-2_arm64.deb"],
        deps = ["@debian12_libgmp10_aarch64//:all_files", "@debian12_libnettle8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libhogweed6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ed8185c28b2cb519744a5a462dcd720d3b332c9b88a1d0002eac06dc8550cb94",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ed8185c28b2cb519744a5a462dcd720d3b332c9b88a1d0002eac06dc8550cb94.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/nettle/libhogweed6_3.8.1-2_amd64.deb"],
        deps = ["@debian12_libgmp10_x86_64//:all_files", "@debian12_libnettle8_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libhwasan0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c64188f901ecc0f2dd5e3ff9de40d3b295d13e495ca13595f3f5a41615de1aef",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c64188f901ecc0f2dd5e3ff9de40d3b295d13e495ca13595f3f5a41615de1aef.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libhwasan0_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files", "@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libicu-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b3bb64f66066e76ad12a528044b39b894a71c717aea6daffb0f126da784456f2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b3bb64f66066e76ad12a528044b39b894a71c717aea6daffb0f126da784456f2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/icu/libicu-dev_72.1-3_arm64.deb"],
        deps = ["@debian12_icu-devtools_aarch64//:all_files", "@debian12_libicu72_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libicu-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "0ba62f3e2ab0af0e31b0292608f539fe50acb3b08ee52438f55558a07ca12958",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0ba62f3e2ab0af0e31b0292608f539fe50acb3b08ee52438f55558a07ca12958.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/icu/libicu-dev_72.1-3_amd64.deb"],
        deps = ["@debian12_icu-devtools_x86_64//:all_files", "@debian12_libicu72_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libicu72_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fa1b61e24b45d07c9ec15dbd1750aeea26eef6044270629ef58138fc09ca238f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fa1b61e24b45d07c9ec15dbd1750aeea26eef6044270629ef58138fc09ca238f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/icu/libicu72_72.1-3_arm64.deb"],
        deps = ["@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libicu72_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e239c1c9f52bee0ff627f291552d63691b765ec7c5cdf6de7c7ae4dec0275857",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e239c1c9f52bee0ff627f291552d63691b765ec7c5cdf6de7c7ae4dec0275857.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/icu/libicu72_72.1-3_amd64.deb"],
        deps = ["@debian12_libgcc-s1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libidn2-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "12a3efc056671bf1c1bed4c3444c2559c8d5e0c158a13316fc728f263b83ddc4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/12a3efc056671bf1c1bed4c3444c2559c8d5e0c158a13316fc728f263b83ddc4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libi/libidn2/libidn2-0_2.3.3-1+b1_arm64.deb"],
        deps = ["@debian12_libunistring2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libidn2-0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d50716d5824083d667427817d506b45d3f59dc77e1ca52de000f3f62d4918afa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d50716d5824083d667427817d506b45d3f59dc77e1ca52de000f3f62d4918afa.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libi/libidn2/libidn2-0_2.3.3-1+b1_amd64.deb"],
        deps = ["@debian12_libunistring2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libip4tc2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fc5b488a30f697d13c8547ca32cebf53880141af77ec36c58b04f899b5708208",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fc5b488a30f697d13c8547ca32cebf53880141af77ec36c58b04f899b5708208.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/iptables/libip4tc2_1.8.9-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libip4tc2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f2c48b367f9ec13f9aa577e7ccf81b371ce5d5fe22dddf9d7aa99f1e0bb7cfc4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f2c48b367f9ec13f9aa577e7ccf81b371ce5d5fe22dddf9d7aa99f1e0bb7cfc4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/iptables/libip4tc2_1.8.9-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libip6tc2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f921206811bbda3dfa03995142ee704fd13f2bb07291ed418bd41fbea3a1d8cc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f921206811bbda3dfa03995142ee704fd13f2bb07291ed418bd41fbea3a1d8cc.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/iptables/libip6tc2_1.8.9-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libip6tc2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "be88db3dfa2fe3345ea343207f7c75345602686121e579023d021c348b9b4f4d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/be88db3dfa2fe3345ea343207f7c75345602686121e579023d021c348b9b4f4d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/iptables/libip6tc2_1.8.9-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libitm1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "68b148cf058da8f361ee1bb3829c3ece1ae318ad956246d78989e0f2d80b4f5b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/68b148cf058da8f361ee1bb3829c3ece1ae318ad956246d78989e0f2d80b4f5b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libitm1_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libitm1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a6b79588938ef738fe6f03582b3ca0ed4fbd4a152dbe9f960e51a0355479a117",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a6b79588938ef738fe6f03582b3ca0ed4fbd4a152dbe9f960e51a0355479a117.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libitm1_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libk5crypto3_aarch64",
        sha256 = "750635caa352c1d1fee6189fd14956568105893ab287ef056729dc746bf36180",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/750635caa352c1d1fee6189fd14956568105893ab287ef056729dc746bf36180.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/krb5/libk5crypto3_1.20.1-2+deb12u1_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libkrb5support0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libk5crypto3_x86_64",
        sha256 = "6a91eee690e6ad2207df3a355fc329a58d8e31bf5ca9a9dd4de8f7a1c812ddc5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6a91eee690e6ad2207df3a355fc329a58d8e31bf5ca9a9dd4de8f7a1c812ddc5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/krb5/libk5crypto3_1.20.1-2+deb12u1_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libkrb5support0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libkeyutils1_aarch64",
        sha256 = "aac46cb6faec4e737502b3c2290b7b02f8ba04e8accd5af7fd07934df0c867b1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/aac46cb6faec4e737502b3c2290b7b02f8ba04e8accd5af7fd07934df0c867b1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/keyutils/libkeyutils1_1.6.3-2_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libkeyutils1_x86_64",
        sha256 = "cfac89e6a7a54ff3c6a4f843310e25efeddaa771baeae470bd98bd588c373563",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cfac89e6a7a54ff3c6a4f843310e25efeddaa771baeae470bd98bd588c373563.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/keyutils/libkeyutils1_1.6.3-2_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libkrb5-3_aarch64",
        sha256 = "7f7a23e637eb7b4cfa1edd9a4cc07c61e6b3b440d74ffdc402e84eef5a7681ff",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7f7a23e637eb7b4cfa1edd9a4cc07c61e6b3b440d74ffdc402e84eef5a7681ff.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/krb5/libkrb5-3_1.20.1-2+deb12u1_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libkeyutils1_aarch64//:all_files", "@debian12_libkrb5support0_aarch64//:all_files", "@debian12_libssl3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libkrb5-3_x86_64",
        sha256 = "03ebdf235600f4a8a6d4fbc7080de0a776b1a701f43c4e9697944757591d7809",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/03ebdf235600f4a8a6d4fbc7080de0a776b1a701f43c4e9697944757591d7809.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/krb5/libkrb5-3_1.20.1-2+deb12u1_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libkeyutils1_x86_64//:all_files", "@debian12_libkrb5support0_x86_64//:all_files", "@debian12_libssl3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libkrb5support0_aarch64",
        sha256 = "aa26a06447e38931ebd4c0a7ce2420b138c63558f4c6b5c736dda573b9deba30",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/aa26a06447e38931ebd4c0a7ce2420b138c63558f4c6b5c736dda573b9deba30.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/krb5/libkrb5support0_1.20.1-2+deb12u1_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libkrb5support0_x86_64",
        sha256 = "e489a9282c4b765c29d9eda7c4747e1cb58be71161012c3a57e2a8bc63dc0f5a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e489a9282c4b765c29d9eda7c4747e1cb58be71161012c3a57e2a8bc63dc0f5a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/k/krb5/libkrb5support0_1.20.1-2+deb12u1_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libksba8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1b76a0b7dd4a24378849ed4e2549e60f975905e2b19134808472ca8c0b82e685",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1b76a0b7dd4a24378849ed4e2549e60f975905e2b19134808472ca8c0b82e685.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libk/libksba/libksba8_1.6.3-2_arm64.deb"],
        deps = ["@debian12_libgpg-error0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libksba8_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b52ffe8f80020a0df90d5fc188561010042ee8a67aae6de463d141a5fc09e1bc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b52ffe8f80020a0df90d5fc188561010042ee8a67aae6de463d141a5fc09e1bc.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libk/libksba/libksba8_1.6.3-2_amd64.deb"],
        deps = ["@debian12_libgpg-error0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libldap-2.5-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "71e6ae4291a20db999c62cd0a4f30a92a799c46b99205d5e787d4acd7c343b91",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/71e6ae4291a20db999c62cd0a4f30a92a799c46b99205d5e787d4acd7c343b91.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/o/openldap/libldap-2.5-0_2.5.13+dfsg-5_arm64.deb"],
        deps = ["@debian12_libsasl2-2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libldap-2.5-0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4b6c30f6554149c594628d945edc6003f0eea8d0cc1341638c0e71375db147ed",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4b6c30f6554149c594628d945edc6003f0eea8d0cc1341638c0e71375db147ed.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/o/openldap/libldap-2.5-0_2.5.13+dfsg-5_amd64.deb"],
        deps = ["@debian12_libsasl2-2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libllvm15_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9eecca617a53fb80ce0de5c6f475817528b3284cd43eec3863126715dc850dd2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9eecca617a53fb80ce0de5c6f475817528b3284cd43eec3863126715dc850dd2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/llvm-toolchain-15/libllvm15_15.0.6-4+b1_arm64.deb"],
        deps = ["@debian12_libedit2_aarch64//:all_files", "@debian12_libffi8_aarch64//:all_files", "@debian12_libgcc-s1_aarch64//:all_files", "@debian12_libxml2_aarch64//:all_files", "@debian12_libz3-4_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libllvm15_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9f0751109ba89e65b1313a4f3e34a29977a0db6fa30ed475e2c6bd555fa9e866",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9f0751109ba89e65b1313a4f3e34a29977a0db6fa30ed475e2c6bd555fa9e866.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/llvm-toolchain-15/libllvm15_15.0.6-4+b1_amd64.deb"],
        deps = ["@debian12_libedit2_x86_64//:all_files", "@debian12_libffi8_x86_64//:all_files", "@debian12_libgcc-s1_x86_64//:all_files", "@debian12_libxml2_x86_64//:all_files", "@debian12_libz3-4_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_liblsan0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ab6f08bdb0db2ae261f5f04af8bfbff9701f97324b30f72ae99463a795246f54",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ab6f08bdb0db2ae261f5f04af8bfbff9701f97324b30f72ae99463a795246f54.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/liblsan0_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files", "@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_liblsan0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c6a494d3605341a2c909e280f81fa015a4c8df2de8624c88a712a7f98a63f057",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c6a494d3605341a2c909e280f81fa015a4c8df2de8624c88a712a7f98a63f057.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/liblsan0_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files", "@debian12_libgcc-s1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_liblz4-1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f061216ce11aabba8f032dfd6c75c181e782fef7493033b9621a8c3b2953b87e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f061216ce11aabba8f032dfd6c75c181e782fef7493033b9621a8c3b2953b87e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lz4/liblz4-1_1.9.4-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_liblz4-1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "64cde86cef1deaf828bd60297839b59710b5cd8dc50efd4f12643caaee9389d3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/64cde86cef1deaf828bd60297839b59710b5cd8dc50efd4f12643caaee9389d3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lz4/liblz4-1_1.9.4-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_liblzma-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "02f19dd7e0f1d1cb7206aa95c81459cc2cabd69f6f5e8a7bc24e58803bea215e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/02f19dd7e0f1d1cb7206aa95c81459cc2cabd69f6f5e8a7bc24e58803bea215e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xz-utils/liblzma-dev_5.4.1-0.2_arm64.deb"],
        deps = ["@debian12_liblzma5_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_liblzma-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6853282a9aaad7f758972a7d52a1bcd9eeb26e224bf9a3d1efb8b3979d42cd4a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6853282a9aaad7f758972a7d52a1bcd9eeb26e224bf9a3d1efb8b3979d42cd4a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xz-utils/liblzma-dev_5.4.1-0.2_amd64.deb"],
        deps = ["@debian12_liblzma5_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_liblzma5_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "48216df0ab15bf757176417c154c27a208b82aa42b00a16794e4699ec9e8e2e3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/48216df0ab15bf757176417c154c27a208b82aa42b00a16794e4699ec9e8e2e3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xz-utils/liblzma5_5.4.1-0.2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_liblzma5_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d4b7736e58512a2b047f9cb91b71db5a3cf9d3451192fc6da044c77bf51fe869",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d4b7736e58512a2b047f9cb91b71db5a3cf9d3451192fc6da044c77bf51fe869.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xz-utils/liblzma5_5.4.1-0.2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmagic-mgc_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ff57ca94d3c1231b16d10587e60a2cf6c53b53249879235c59fa0b19e832974c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ff57ca94d3c1231b16d10587e60a2cf6c53b53249879235c59fa0b19e832974c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/f/file/libmagic-mgc_5.44-3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmagic-mgc_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "bad01673ba5dfb9b5db4f3ae6a71f18d492cb6801eab45ad3c7d483c0a1f6ad2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bad01673ba5dfb9b5db4f3ae6a71f18d492cb6801eab45ad3c7d483c0a1f6ad2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/f/file/libmagic-mgc_5.44-3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmagic1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2fbaa1e2efdb80cf20cb13b2c523e82602c117cf5ec922c6d1d4ce6e18440052",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2fbaa1e2efdb80cf20cb13b2c523e82602c117cf5ec922c6d1d4ce6e18440052.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/f/file/libmagic1_5.44-3_arm64.deb"],
        deps = ["@debian12_libbz2-1.0_aarch64//:all_files", "@debian12_liblzma5_aarch64//:all_files", "@debian12_libmagic-mgc_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmagic1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a241c2adc7438a7e217f32544028489981768a349d3e48673392703255c7b88e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a241c2adc7438a7e217f32544028489981768a349d3e48673392703255c7b88e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/f/file/libmagic1_5.44-3_amd64.deb"],
        deps = ["@debian12_libbz2-1.0_x86_64//:all_files", "@debian12_liblzma5_x86_64//:all_files", "@debian12_libmagic-mgc_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmd0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "99a8c7dd591fae9fb37d8bf8dfdffa850e207fa405b3198c5b24711a5f972381",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/99a8c7dd591fae9fb37d8bf8dfdffa850e207fa405b3198c5b24711a5f972381.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libm/libmd/libmd0_1.0.4-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmd0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "03539fd30c509e27101d13a56e52eda9062bdf1aefe337c07ab56def25a13eab",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/03539fd30c509e27101d13a56e52eda9062bdf1aefe337c07ab56def25a13eab.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libm/libmd/libmd0_1.0.4-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmnl0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d63aafb6f2c07db8fcb135b00ff915baf72ef8a3397e773c9c24d67950c6a46c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d63aafb6f2c07db8fcb135b00ff915baf72ef8a3397e773c9c24d67950c6a46c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libm/libmnl/libmnl0_1.0.4-3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmnl0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4581f42e3373cb72f9ea4e88163b17873afca614a6c6f54637e95aa75983ea7c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4581f42e3373cb72f9ea4e88163b17873afca614a6c6f54637e95aa75983ea7c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libm/libmnl/libmnl0_1.0.4-3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmount1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "230893a5d0e6d1fb3cbf97d46139d10421c618f4fb23708ca141520c05b94d9e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/230893a5d0e6d1fb3cbf97d46139d10421c618f4fb23708ca141520c05b94d9e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/u/util-linux/libmount1_2.38.1-5+b1_arm64.deb"],
        deps = ["@debian12_libblkid1_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmount1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8a2f81076419cd6b0def5cd1fac98383c85ddec1a5c388f57e8e9e2fdf491ad9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8a2f81076419cd6b0def5cd1fac98383c85ddec1a5c388f57e8e9e2fdf491ad9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/u/util-linux/libmount1_2.38.1-5+b1_amd64.deb"],
        deps = ["@debian12_libblkid1_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmpfr6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d818fbfb555561bff56694cc628dc9939f393a07026aefd237e9c818b9f2cb77",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d818fbfb555561bff56694cc628dc9939f393a07026aefd237e9c818b9f2cb77.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mpfr4/libmpfr6_4.2.0-1_arm64.deb"],
        deps = ["@debian12_libgmp10_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libmpfr6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "37b7a2b4e78890b6a074777f27b96c84f58e81558ba08410c2b6c0ca4a4ad77b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/37b7a2b4e78890b6a074777f27b96c84f58e81558ba08410c2b6c0ca4a4ad77b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mpfr4/libmpfr6_4.2.0-1_amd64.deb"],
        deps = ["@debian12_libgmp10_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libncurses-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a33e3269e924132dd49b4424754ca63ae2b901a89bc8a3f541e962d67bee9d41",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a33e3269e924132dd49b4424754ca63ae2b901a89bc8a3f541e962d67bee9d41.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/libncurses-dev_6.4-4_arm64.deb"],
        deps = ["@debian12_libc6-dev_aarch64//:all_files", "@debian12_libncurses6_aarch64//:all_files", "@debian12_libncursesw6_aarch64//:all_files", "@debian12_ncurses-bin_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libncurses-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2ad228835756feb118bb131b32834bd23a09047e4de408cc5204cbb5dce0e4bb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2ad228835756feb118bb131b32834bd23a09047e4de408cc5204cbb5dce0e4bb.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/libncurses-dev_6.4-4_amd64.deb"],
        deps = ["@debian12_libc6-dev_x86_64//:all_files", "@debian12_libncurses6_x86_64//:all_files", "@debian12_libncursesw6_x86_64//:all_files", "@debian12_ncurses-bin_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libncurses6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "bb5013fc7f24f3ed740d50449bfea3fb4ffce1e33787b6bc7f82ea3d377bc03c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bb5013fc7f24f3ed740d50449bfea3fb4ffce1e33787b6bc7f82ea3d377bc03c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/libncurses6_6.4-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libncurses6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "72300f09f02669c06c99b641ea795d52300ec7eb65eaccddf7bc3b72934f0ef5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/72300f09f02669c06c99b641ea795d52300ec7eb65eaccddf7bc3b72934f0ef5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/libncurses6_6.4-4_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libncursesw6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "cf32cb6751718872c6def448b82211eec494f688e2f1a3e6c71bfdaf6b0722c5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cf32cb6751718872c6def448b82211eec494f688e2f1a3e6c71bfdaf6b0722c5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/libncursesw6_6.4-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libncursesw6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "98fa7a53dc565a38b65fb70422ad08001bf5361d8fbc74255280c329996a6bec",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/98fa7a53dc565a38b65fb70422ad08001bf5361d8fbc74255280c329996a6bec.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/libncursesw6_6.4-4_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnetfilter-conntrack3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "837ebd84c43dc9f6c840695f64763ff4a3b7e5e83eedb37817e272e75fcc84bf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/837ebd84c43dc9f6c840695f64763ff4a3b7e5e83eedb37817e272e75fcc84bf.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnetfilter-conntrack/libnetfilter-conntrack3_1.0.9-3_arm64.deb"],
        deps = ["@debian12_libnfnetlink0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnetfilter-conntrack3_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "65f1539238f60fcc85d115acd640474f45c77bfddcf402eb7d75965a783c2bc8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/65f1539238f60fcc85d115acd640474f45c77bfddcf402eb7d75965a783c2bc8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnetfilter-conntrack/libnetfilter-conntrack3_1.0.9-3_amd64.deb"],
        deps = ["@debian12_libnfnetlink0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnettle8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c945ff210df69cf7b95e935b8fa936e81c1c1f475355e3d5db83510b174f0cd6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c945ff210df69cf7b95e935b8fa936e81c1c1f475355e3d5db83510b174f0cd6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/nettle/libnettle8_3.8.1-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnettle8_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "45922e6e289ffd92f0f92d2bb9159e84236ff202d552a461bf10e5335b3f0261",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/45922e6e289ffd92f0f92d2bb9159e84236ff202d552a461bf10e5335b3f0261.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/nettle/libnettle8_3.8.1-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnfnetlink0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e2c85edf9015dc918ce190397fbbf059ad659941c5a650a250de67f2319a903b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e2c85edf9015dc918ce190397fbbf059ad659941c5a650a250de67f2319a903b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnfnetlink/libnfnetlink0_1.0.2-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnfnetlink0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ae5cbba417ea48f34c1f72c27e8146a81f20614c1296bca2cd7234c8215fddcc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ae5cbba417ea48f34c1f72c27e8146a81f20614c1296bca2cd7234c8215fddcc.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnfnetlink/libnfnetlink0_1.0.2-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnftnl11_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fd8b6f2ea21d60e25b07dfee8804ece79c2b18309df7773b091a587135565b9f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fd8b6f2ea21d60e25b07dfee8804ece79c2b18309df7773b091a587135565b9f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnftnl/libnftnl11_1.2.4-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnftnl11_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9e619e1470a1915264e906020f3bfc046fd8458043b1342686d997b5078213af",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9e619e1470a1915264e906020f3bfc046fd8458043b1342686d997b5078213af.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnftnl/libnftnl11_1.2.4-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnpth0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5499a10044afad14685bbfc4026ac62630562da08d5b9a15a546c6b62bd6ea89",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5499a10044afad14685bbfc4026ac62630562da08d5b9a15a546c6b62bd6ea89.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/npth/libnpth0_1.6-3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnpth0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "43c90d45f7cf5584108964b919d6c728680d81af5fa70c8fb367d661cef54e8c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/43c90d45f7cf5584108964b919d6c728680d81af5fa70c8fb367d661cef54e8c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/npth/libnpth0_1.6-3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnsl2_aarch64",
        sha256 = "8f9ba58b219779b43c4ccc78c79b0a23f721fc96323c202abb31e02f942104b3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8f9ba58b219779b43c4ccc78c79b0a23f721fc96323c202abb31e02f942104b3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnsl/libnsl2_1.3.0-2_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libtirpc3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnsl2_x86_64",
        sha256 = "c0d83437fdb016cb289436f49f28a36be44b3e8f1f2498c7e3a095f709c0d6f8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c0d83437fdb016cb289436f49f28a36be44b3e8f1f2498c7e3a095f709c0d6f8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnsl/libnsl2_1.3.0-2_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libtirpc3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnumber-compare-perl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "cf95662e27037d92c582647400bb1ca7a6dfd2eb26eee5b968b23263cd4976dc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cf95662e27037d92c582647400bb1ca7a6dfd2eb26eee5b968b23263cd4976dc.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnumber-compare-perl/libnumber-compare-perl_0.03-3_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libnumber-compare-perl_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "cf95662e27037d92c582647400bb1ca7a6dfd2eb26eee5b968b23263cd4976dc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cf95662e27037d92c582647400bb1ca7a6dfd2eb26eee5b968b23263cd4976dc.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libn/libnumber-compare-perl/libnumber-compare-perl_0.03-3_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libopengl-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b4ddc2d38103e9f0c3637d46334ccbf40c30e851c85f5348abc0aa37e3926822",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b4ddc2d38103e9f0c3637d46334ccbf40c30e851c85f5348abc0aa37e3926822.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libopengl-dev_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libopengl0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libopengl-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "43d5cd7de2f591e27f95996f8484767f46b0bbc1b8a0ac58cc64987b99fb9369",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/43d5cd7de2f591e27f95996f8484767f46b0bbc1b8a0ac58cc64987b99fb9369.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libopengl-dev_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libopengl0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libopengl0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4b3c6b3bc25530290cf231acac4c85b8806dda53ab6376ab22b3ec511b4fb7a0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4b3c6b3bc25530290cf231acac4c85b8806dda53ab6376ab22b3ec511b4fb7a0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libopengl0_1.6.0-1_arm64.deb"],
        deps = ["@debian12_libglvnd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libopengl0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "dc71cb5aaddeb8b09b3d63b8426fd651d8c79ad55b23dbe640c1abbc94c85013",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/dc71cb5aaddeb8b09b3d63b8426fd651d8c79ad55b23dbe640c1abbc94c85013.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libg/libglvnd/libopengl0_1.6.0-1_amd64.deb"],
        deps = ["@debian12_libglvnd0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libp11-kit0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d1f1f55023e9fc085b9ebfc9c4113d2d2dab2dc6b81a337f274b75c95ad8dc0a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d1f1f55023e9fc085b9ebfc9c4113d2d2dab2dc6b81a337f274b75c95ad8dc0a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/p11-kit/libp11-kit0_0.24.1-2_arm64.deb"],
        deps = ["@debian12_libffi8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libp11-kit0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "251330faddbf013f060fcdb41f4b0c037c8a6e89ba7c09b04bfcc4e3f0807b22",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/251330faddbf013f060fcdb41f4b0c037c8a6e89ba7c09b04bfcc4e3f0807b22.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/p11-kit/libp11-kit0_0.24.1-2_amd64.deb"],
        deps = ["@debian12_libffi8_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpam-modules-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f89db9843fa29e7571687156166ebb8753644105bc3c4e35ac165ca5c9c740b1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f89db9843fa29e7571687156166ebb8753644105bc3c4e35ac165ca5c9c740b1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pam/libpam-modules-bin_1.5.2-6+deb12u1_arm64.deb"],
        deps = ["@debian12_libaudit1_aarch64//:all_files", "@debian12_libcrypt1_aarch64//:all_files", "@debian12_libpam0g_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpam-modules-bin_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fc6a692d2f399b83ef5a7f310883286a5e4326095812d8bb934925125002981c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fc6a692d2f399b83ef5a7f310883286a5e4326095812d8bb934925125002981c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pam/libpam-modules-bin_1.5.2-6+deb12u1_amd64.deb"],
        deps = ["@debian12_libaudit1_x86_64//:all_files", "@debian12_libcrypt1_x86_64//:all_files", "@debian12_libpam0g_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpam-modules_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f2acfa1766b31a2b2b89c7afe11f757c88ec2f1d8abc8ce5bd77c4dd3e5fa24b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f2acfa1766b31a2b2b89c7afe11f757c88ec2f1d8abc8ce5bd77c4dd3e5fa24b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pam/libpam-modules_1.5.2-6+deb12u1_arm64.deb"],
        deps = ["@debian12_debconf_aarch64//:all_files", "@debian12_libaudit1_aarch64//:all_files", "@debian12_libcrypt1_aarch64//:all_files", "@debian12_libdb5.3_aarch64//:all_files", "@debian12_libpam-modules-bin_aarch64//:all_files", "@debian12_libpam0g_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpam-modules_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "851d270e36707787ab1cd269dbd9597864feaf3f8453ecd3c426caaa56142222",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/851d270e36707787ab1cd269dbd9597864feaf3f8453ecd3c426caaa56142222.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pam/libpam-modules_1.5.2-6+deb12u1_amd64.deb"],
        deps = ["@debian12_debconf_x86_64//:all_files", "@debian12_libaudit1_x86_64//:all_files", "@debian12_libcrypt1_x86_64//:all_files", "@debian12_libdb5.3_x86_64//:all_files", "@debian12_libpam-modules-bin_x86_64//:all_files", "@debian12_libpam0g_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpam0g_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "7432311323e7648a7104ddc3332d994600cf0d2a1bd4e2a44679b06e9d932eba",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7432311323e7648a7104ddc3332d994600cf0d2a1bd4e2a44679b06e9d932eba.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pam/libpam0g_1.5.2-6+deb12u1_arm64.deb"],
        deps = ["@debian12_debconf_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpam0g_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e360be5f17f9c09c8f17bae809f6c6f091c5bb6ab1a44fc33e4fb86c5e5559df",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e360be5f17f9c09c8f17bae809f6c6f091c5bb6ab1a44fc33e4fb86c5e5559df.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pam/libpam0g_1.5.2-6+deb12u1_amd64.deb"],
        deps = ["@debian12_debconf_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpciaccess-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a87c2ace4c1eedb79188303c3938c6bb485c76d9ee5931f61f90d94b6fa7ea18",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a87c2ace4c1eedb79188303c3938c6bb485c76d9ee5931f61f90d94b6fa7ea18.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libp/libpciaccess/libpciaccess-dev_0.17-2_arm64.deb"],
        deps = ["@debian12_libpciaccess0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpciaccess-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3878f504d4060229c2fe0c1e573a7fdbe7f55744c948ce04e214ddb4a465e5cc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3878f504d4060229c2fe0c1e573a7fdbe7f55744c948ce04e214ddb4a465e5cc.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libp/libpciaccess/libpciaccess-dev_0.17-2_amd64.deb"],
        deps = ["@debian12_libpciaccess0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpciaccess0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3caffef1ee26d9f8699b204bc1f9e34c5c08e147a5dad5da3ca4d1776918e905",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3caffef1ee26d9f8699b204bc1f9e34c5c08e147a5dad5da3ca4d1776918e905.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libp/libpciaccess/libpciaccess0_0.17-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpciaccess0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4025f3608cf431c163efb94fdc553e7b93e16b8f0d741ea87762e19025ffc80e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4025f3608cf431c163efb94fdc553e7b93e16b8f0d741ea87762e19025ffc80e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libp/libpciaccess/libpciaccess0_0.17-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpcre2-8-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b2448d0a8a3db7fbeac231e7ef93811346c1fb5f96ccf6f631701d8a4eb39206",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b2448d0a8a3db7fbeac231e7ef93811346c1fb5f96ccf6f631701d8a4eb39206.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pcre2/libpcre2-8-0_10.42-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpcre2-8-0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "030db54f4d76cdfe2bf0e8eb5f9efea0233ab3c7aa942d672c7b63b52dbaf935",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/030db54f4d76cdfe2bf0e8eb5f9efea0233ab3c7aa942d672c7b63b52dbaf935.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pcre2/libpcre2-8-0_10.42-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libperl5.36_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "73be5c822b41786010754ba04e496e5b02b54c03b663046c14675d826de701b5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/73be5c822b41786010754ba04e496e5b02b54c03b663046c14675d826de701b5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/perl/libperl5.36_5.36.0-7+deb12u1_arm64.deb"],
        deps = ["@debian12_libbz2-1.0_aarch64//:all_files", "@debian12_libcrypt1_aarch64//:all_files", "@debian12_libdb5.3_aarch64//:all_files", "@debian12_libgdbm-compat4_aarch64//:all_files", "@debian12_libgdbm6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libperl5.36_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4b48b8f0b06c2c667d52117edcef69af6896bcfe69a4f4bde47b89590b83875e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4b48b8f0b06c2c667d52117edcef69af6896bcfe69a4f4bde47b89590b83875e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/perl/libperl5.36_5.36.0-7+deb12u1_amd64.deb"],
        deps = ["@debian12_libbz2-1.0_x86_64//:all_files", "@debian12_libcrypt1_x86_64//:all_files", "@debian12_libdb5.3_x86_64//:all_files", "@debian12_libgdbm-compat4_x86_64//:all_files", "@debian12_libgdbm6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpthread-stubs0-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "acd58ed5d26d3f4d4b0856ebec6bc43f9c4357601a532eee0d2de839a8b8952e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/acd58ed5d26d3f4d4b0856ebec6bc43f9c4357601a532eee0d2de839a8b8952e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libp/libpthread-stubs/libpthread-stubs0-dev_0.4-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpthread-stubs0-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "54632f160e1e8a43656a87195a547391038c4ca0f53291b849cd4457ba5dfde9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/54632f160e1e8a43656a87195a547391038c4ca0f53291b849cd4457ba5dfde9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libp/libpthread-stubs/libpthread-stubs0-dev_0.4-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpython3.11-minimal_aarch64",
        sha256 = "399271d141a1e08479c80f7832921ddf5fdb46c15c8b0dd3172c52c74b79b197",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/399271d141a1e08479c80f7832921ddf5fdb46c15c8b0dd3172c52c74b79b197.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/python3.11/libpython3.11-minimal_3.11.2-6_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libssl3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpython3.11-minimal_x86_64",
        sha256 = "6fc5ab5858781ab90c68b4deea09f21871fd7b55dc1a0764ad7116ac4c86574d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6fc5ab5858781ab90c68b4deea09f21871fd7b55dc1a0764ad7116ac4c86574d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/python3.11/libpython3.11-minimal_3.11.2-6_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libssl3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpython3.11-stdlib_aarch64",
        sha256 = "1af0700762fc1ca17000e124a456e50f6a769808ea1a6a0d0e941ec87647fb3a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1af0700762fc1ca17000e124a456e50f6a769808ea1a6a0d0e941ec87647fb3a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/python3.11/libpython3.11-stdlib_3.11.2-6_arm64.deb"],
        deps = ["@debian12_libbz2-1.0_aarch64//:all_files", "@debian12_libc6_aarch64//:all_files", "@debian12_libcrypt1_aarch64//:all_files", "@debian12_libdb5.3_aarch64//:all_files", "@debian12_libffi8_aarch64//:all_files", "@debian12_liblzma5_aarch64//:all_files", "@debian12_libncursesw6_aarch64//:all_files", "@debian12_libnsl2_aarch64//:all_files", "@debian12_libpython3.11-minimal_aarch64//:all_files", "@debian12_libreadline8_aarch64//:all_files", "@debian12_libsqlite3-0_aarch64//:all_files", "@debian12_libtinfo6_aarch64//:all_files", "@debian12_libtirpc3_aarch64//:all_files", "@debian12_libuuid1_aarch64//:all_files", "@debian12_media-types_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libpython3.11-stdlib_x86_64",
        sha256 = "67eec0eb4df58b93e1bf97c402c2cbeb361bf9c5af44fa3a02ff1c723c791ca2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/67eec0eb4df58b93e1bf97c402c2cbeb361bf9c5af44fa3a02ff1c723c791ca2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/python3.11/libpython3.11-stdlib_3.11.2-6_amd64.deb"],
        deps = ["@debian12_libbz2-1.0_x86_64//:all_files", "@debian12_libc6_x86_64//:all_files", "@debian12_libcrypt1_x86_64//:all_files", "@debian12_libdb5.3_x86_64//:all_files", "@debian12_libffi8_x86_64//:all_files", "@debian12_liblzma5_x86_64//:all_files", "@debian12_libncursesw6_x86_64//:all_files", "@debian12_libnsl2_x86_64//:all_files", "@debian12_libpython3.11-minimal_x86_64//:all_files", "@debian12_libreadline8_x86_64//:all_files", "@debian12_libsqlite3-0_x86_64//:all_files", "@debian12_libtinfo6_x86_64//:all_files", "@debian12_libtirpc3_x86_64//:all_files", "@debian12_libuuid1_x86_64//:all_files", "@debian12_media-types_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libquadmath0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4e21728bbb1f170f35a5d60fe26adadb48c436f1b5fd977454e632668074169c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4e21728bbb1f170f35a5d60fe26adadb48c436f1b5fd977454e632668074169c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libquadmath0_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libreadline8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f3b162b7c1e05430607e792ebdbfc417cbd1f1d32cf83664133ae63d811a72d2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f3b162b7c1e05430607e792ebdbfc417cbd1f1d32cf83664133ae63d811a72d2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/r/readline/libreadline8_8.2-1.3_arm64.deb"],
        deps = ["@debian12_readline-common_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libreadline8_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e02ebbd3701cf468dbf98d6d917fbe0325e881f07fe8b316150c8d2a64486e66",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e02ebbd3701cf468dbf98d6d917fbe0325e881f07fe8b316150c8d2a64486e66.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/r/readline/libreadline8_8.2-1.3_amd64.deb"],
        deps = ["@debian12_readline-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsasl2-2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "61dcbb6560e2eb20bff256f1b445ac9af13aa61c6a6ae115ad8cb96c9c50ea38",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/61dcbb6560e2eb20bff256f1b445ac9af13aa61c6a6ae115ad8cb96c9c50ea38.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/cyrus-sasl2/libsasl2-2_2.1.28+dfsg-10_arm64.deb"],
        deps = ["@debian12_libsasl2-modules-db_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsasl2-2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "11ee190ad39f8d7af441d2c8347388b9449434c73acc67b4b372445ac4152efa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/11ee190ad39f8d7af441d2c8347388b9449434c73acc67b4b372445ac4152efa.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/cyrus-sasl2/libsasl2-2_2.1.28+dfsg-10_amd64.deb"],
        deps = ["@debian12_libsasl2-modules-db_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsasl2-modules-db_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "56d9c35ac6729b02f78900175557dd36bef26611dab89584f29da15631628869",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/56d9c35ac6729b02f78900175557dd36bef26611dab89584f29da15631628869.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/cyrus-sasl2/libsasl2-modules-db_2.1.28+dfsg-10_arm64.deb"],
        deps = ["@debian12_libdb5.3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsasl2-modules-db_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3ac4fd6cbe3b3b06e68d24b931bf3eb9385b42f15604a37ed25310e948ca0ee6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3ac4fd6cbe3b3b06e68d24b931bf3eb9385b42f15604a37ed25310e948ca0ee6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/c/cyrus-sasl2/libsasl2-modules-db_2.1.28+dfsg-10_amd64.deb"],
        deps = ["@debian12_libdb5.3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libseccomp2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "bd820298b54d27284844bf7c10920e0bedaf57d2565abf9636a7f4a245255c40",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bd820298b54d27284844bf7c10920e0bedaf57d2565abf9636a7f4a245255c40.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libseccomp/libseccomp2_2.5.4-1+b3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libseccomp2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9e6305a100f5178cc321ee33b96933a6482d11fdc22b42c0e526d6151c0c6f0f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9e6305a100f5178cc321ee33b96933a6482d11fdc22b42c0e526d6151c0c6f0f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libseccomp/libseccomp2_2.5.4-1+b3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libselinux1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "29201edf23ebae40844d6c289afdb9bba52f927d55096ed1b1cd37e040135edc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/29201edf23ebae40844d6c289afdb9bba52f927d55096ed1b1cd37e040135edc.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libselinux/libselinux1_3.4-1+b6_arm64.deb"],
        deps = ["@debian12_libpcre2-8-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libselinux1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2b07f5287b9105f40158b56e4d70cc1652dac56a408f3507b4ab3d061eed425f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2b07f5287b9105f40158b56e4d70cc1652dac56a408f3507b4ab3d061eed425f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libselinux/libselinux1_3.4-1+b6_amd64.deb"],
        deps = ["@debian12_libpcre2-8-0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsemanage-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "835f806c21ae25e39053bd3057051640341b0cf08e1db9746fd82e370d82fa30",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/835f806c21ae25e39053bd3057051640341b0cf08e1db9746fd82e370d82fa30.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libsemanage/libsemanage-common_3.4-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsemanage-common_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "835f806c21ae25e39053bd3057051640341b0cf08e1db9746fd82e370d82fa30",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/835f806c21ae25e39053bd3057051640341b0cf08e1db9746fd82e370d82fa30.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libsemanage/libsemanage-common_3.4-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsemanage2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6d1958b29ac622d352e00f9de55d9de8aea12bd0c27dee8b522e052ace3c67bd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6d1958b29ac622d352e00f9de55d9de8aea12bd0c27dee8b522e052ace3c67bd.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libsemanage/libsemanage2_3.4-1+b5_arm64.deb"],
        deps = ["@debian12_libbz2-1.0_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files", "@debian12_libsemanage-common_aarch64//:all_files", "@debian12_libsepol2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsemanage2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fd36d0972866adde5a52269a309fcecd76a8e45e557dd0ecd33aa221cabc2a8c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fd36d0972866adde5a52269a309fcecd76a8e45e557dd0ecd33aa221cabc2a8c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libsemanage/libsemanage2_3.4-1+b5_amd64.deb"],
        deps = ["@debian12_libbz2-1.0_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files", "@debian12_libsemanage-common_x86_64//:all_files", "@debian12_libsepol2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsensors-config_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "7f3c9fbd822858a9e30335e4a7f66c9468962eb26cd375b93bc8b789660bf02f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7f3c9fbd822858a9e30335e4a7f66c9468962eb26cd375b93bc8b789660bf02f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lm-sensors/libsensors-config_3.6.0-7.1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsensors-config_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "7f3c9fbd822858a9e30335e4a7f66c9468962eb26cd375b93bc8b789660bf02f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7f3c9fbd822858a9e30335e4a7f66c9468962eb26cd375b93bc8b789660bf02f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lm-sensors/libsensors-config_3.6.0-7.1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsensors5_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "adfdf7c7ee1a1c7bc61b5bf691d3a501e857c44243c7cdb6c8b81de86592e03b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/adfdf7c7ee1a1c7bc61b5bf691d3a501e857c44243c7cdb6c8b81de86592e03b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lm-sensors/libsensors5_3.6.0-7.1_arm64.deb"],
        deps = ["@debian12_libsensors-config_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsensors5_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b7eb91dce728fbb9203aec8b22637303b29821c3384e5f78a8ff348b4e44efe3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b7eb91dce728fbb9203aec8b22637303b29821c3384e5f78a8ff348b4e44efe3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lm-sensors/libsensors5_3.6.0-7.1_amd64.deb"],
        deps = ["@debian12_libsensors-config_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsepol2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "22b0041a04af364f643ff2e7ff88eaaecdf0714dcfd253e8c99a6a952ae1fec6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/22b0041a04af364f643ff2e7ff88eaaecdf0714dcfd253e8c99a6a952ae1fec6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libsepol/libsepol2_3.4-2.1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsepol2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b998946bb9818a97b387a962826caae33bc7fdcb6d706b2782c0470510be6b48",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b998946bb9818a97b387a962826caae33bc7fdcb6d706b2782c0470510be6b48.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libsepol/libsepol2_3.4-2.1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libset-scalar-perl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "932ba032b5188db0fed59c23f52bb977f7df5061ff543e7e6bdf303f1559ecaf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/932ba032b5188db0fed59c23f52bb977f7df5061ff543e7e6bdf303f1559ecaf.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libset-scalar-perl/libset-scalar-perl_1.29-3_all.deb"],
        deps = ["@debian12_perl_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libset-scalar-perl_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "932ba032b5188db0fed59c23f52bb977f7df5061ff543e7e6bdf303f1559ecaf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/932ba032b5188db0fed59c23f52bb977f7df5061ff543e7e6bdf303f1559ecaf.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libset-scalar-perl/libset-scalar-perl_1.29-3_all.deb"],
        deps = ["@debian12_perl_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsigsegv2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "455bace82a319cda15e6b1f1bdc0c1ae6e0749bff9a93e09055bc3fd168a2df5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/455bace82a319cda15e6b1f1bdc0c1ae6e0749bff9a93e09055bc3fd168a2df5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libsigsegv/libsigsegv2_2.14-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsigsegv2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "92336a5c47989b82ba4c778adc0ea22fc84450773f2d8428fca4448b0789cfe6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/92336a5c47989b82ba4c778adc0ea22fc84450773f2d8428fca4448b0789cfe6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libs/libsigsegv/libsigsegv2_2.14-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsqlite3-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "87e891926ba874c9f5fbd3b48d564cf103dd97db78f2d83fd1175826771dacfd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/87e891926ba874c9f5fbd3b48d564cf103dd97db78f2d83fd1175826771dacfd.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/sqlite3/libsqlite3-0_3.40.1-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsqlite3-0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a8b11a1664a998cc2499fb04327d1f6c4e8f77b78ea8b6f8418d96fc54e3731f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a8b11a1664a998cc2499fb04327d1f6c4e8f77b78ea8b6f8418d96fc54e3731f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/sqlite3/libsqlite3-0_3.40.1-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libssl3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4e05dfa7319f72c5598f29a39eaf568203e2ae2314d5ef434dbf764de6448462",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4e05dfa7319f72c5598f29a39eaf568203e2ae2314d5ef434dbf764de6448462.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/o/openssl/libssl3_3.0.11-1~deb12u2_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libssl3_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6e129c5814812b3516a656ae5b664b9970e2f8823250cd5b98190f21c0de2bca",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6e129c5814812b3516a656ae5b664b9970e2f8823250cd5b98190f21c0de2bca.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/o/openssl/libssl3_3.0.11-1~deb12u2_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libstdc__-12-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "bc901784391be6a11f9af13c993514e3d8cceb2f584dbee0f59d97589e74251e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bc901784391be6a11f9af13c993514e3d8cceb2f584dbee0f59d97589e74251e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libstdc++-12-dev_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libstdc__-12-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a0f0f3fbeb661d9bda139a54f4bd1c30aa66cd55a8fa0beb0e6bc7946e243ca1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a0f0f3fbeb661d9bda139a54f4bd1c30aa66cd55a8fa0beb0e6bc7946e243ca1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libstdc++-12-dev_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libstdc__6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "21e971c5d3506f783b89efe8e12ac85081ddd9213e4f6529262bcfe95c326670",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/21e971c5d3506f783b89efe8e12ac85081ddd9213e4f6529262bcfe95c326670.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libstdc++6_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files", "@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libstdc__6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9b1b269020cec6aced3b39f096f7b67edd1f0d4ab24f412cb6506d0800e19cbf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9b1b269020cec6aced3b39f096f7b67edd1f0d4ab24f412cb6506d0800e19cbf.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libstdc++6_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files", "@debian12_libgcc-s1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsubid4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "2dcb2cadec95e11d2dfe8f51acb848fbae9b51c197b62c91bd9b104ca4c18992",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2dcb2cadec95e11d2dfe8f51acb848fbae9b51c197b62c91bd9b104ca4c18992.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/shadow/libsubid4_4.13+dfsg1-1+b1_arm64.deb"],
        deps = ["@debian12_libaudit1_aarch64//:all_files", "@debian12_libcrypt1_aarch64//:all_files", "@debian12_libpam0g_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files", "@debian12_libsemanage2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsubid4_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8ef0ee80fe660e5eb5b4640d2f78eb3520e29ddee368dd3e66bd98152052f245",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8ef0ee80fe660e5eb5b4640d2f78eb3520e29ddee368dd3e66bd98152052f245.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/shadow/libsubid4_4.13+dfsg1-1+b1_amd64.deb"],
        deps = ["@debian12_libaudit1_x86_64//:all_files", "@debian12_libcrypt1_x86_64//:all_files", "@debian12_libpam0g_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files", "@debian12_libsemanage2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsystemd0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1afaa32a4c10640b4681275dde0ec21c11f6747674aa0f87085bb65162a5b02d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1afaa32a4c10640b4681275dde0ec21c11f6747674aa0f87085bb65162a5b02d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/systemd/libsystemd0_252.19-1~deb12u1_arm64.deb"],
        deps = ["@debian12_libgcrypt20_aarch64//:all_files", "@debian12_liblz4-1_aarch64//:all_files", "@debian12_liblzma5_aarch64//:all_files", "@debian12_libzstd1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libsystemd0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "339abb97957695134f9df48dfa3eb7df5f681c3aa76a53934133dee2f451d1e4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/339abb97957695134f9df48dfa3eb7df5f681c3aa76a53934133dee2f451d1e4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/systemd/libsystemd0_252.19-1~deb12u1_amd64.deb"],
        deps = ["@debian12_libgcrypt20_x86_64//:all_files", "@debian12_liblz4-1_x86_64//:all_files", "@debian12_liblzma5_x86_64//:all_files", "@debian12_libzstd1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtasn1-6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "235e0097ecf3742ebea01691ce1b01b5504b5de205734dab4a5353f0c324f3f3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/235e0097ecf3742ebea01691ce1b01b5504b5de205734dab4a5353f0c324f3f3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libt/libtasn1-6/libtasn1-6_4.19.0-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtasn1-6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "eec4dc9d949d2c666b1da3fa762a340e8ba10c3a04d3eed32749a97695c15641",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/eec4dc9d949d2c666b1da3fa762a340e8ba10c3a04d3eed32749a97695c15641.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libt/libtasn1-6/libtasn1-6_4.19.0-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtext-glob-perl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b0f1a40c5b26f4e4c83a4f8e1768fcf50ae9aac84b7f807ccb38123b50396f64",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b0f1a40c5b26f4e4c83a4f8e1768fcf50ae9aac84b7f807ccb38123b50396f64.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libt/libtext-glob-perl/libtext-glob-perl_0.11-3_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtext-glob-perl_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b0f1a40c5b26f4e4c83a4f8e1768fcf50ae9aac84b7f807ccb38123b50396f64",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b0f1a40c5b26f4e4c83a4f8e1768fcf50ae9aac84b7f807ccb38123b50396f64.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libt/libtext-glob-perl/libtext-glob-perl_0.11-3_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtinfo6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "baef0f6776f84c7eed4f1146d6e5774689567dad43216894d41da02e6608e4b3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/baef0f6776f84c7eed4f1146d6e5774689567dad43216894d41da02e6608e4b3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/libtinfo6_6.4-4_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtinfo6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "072d908f38f51090ca28ca5afa3b46b2957dc61fe35094c0b851426859a49a51",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/072d908f38f51090ca28ca5afa3b46b2957dc61fe35094c0b851426859a49a51.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/libtinfo6_6.4-4_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtirpc-common_aarch64",
        sha256 = "3e3ef129b4bf61513144236e15e1b4ec57fa5ae3dc8a72137abdbefb7a63af85",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3e3ef129b4bf61513144236e15e1b4ec57fa5ae3dc8a72137abdbefb7a63af85.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libt/libtirpc/libtirpc-common_1.3.3+ds-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtirpc-common_x86_64",
        sha256 = "3e3ef129b4bf61513144236e15e1b4ec57fa5ae3dc8a72137abdbefb7a63af85",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3e3ef129b4bf61513144236e15e1b4ec57fa5ae3dc8a72137abdbefb7a63af85.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libt/libtirpc/libtirpc-common_1.3.3+ds-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtirpc3_aarch64",
        sha256 = "deaf9ea54f74cab6690b4bc0b427f05cbe62cc073580d84f5f18a8d9e2ece401",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/deaf9ea54f74cab6690b4bc0b427f05cbe62cc073580d84f5f18a8d9e2ece401.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libt/libtirpc/libtirpc3_1.3.3+ds-1_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libgssapi-krb5-2_aarch64//:all_files", "@debian12_libtirpc-common_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtirpc3_x86_64",
        sha256 = "2a46d5a5e9486da11ffeff5740931740d6deae4f92cd6098df060dc5dff1e1c7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2a46d5a5e9486da11ffeff5740931740d6deae4f92cd6098df060dc5dff1e1c7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libt/libtirpc/libtirpc3_1.3.3+ds-1_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libgssapi-krb5-2_x86_64//:all_files", "@debian12_libtirpc-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtsan2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "38c4aca9ef3a0301937a6786f0836e7e5e498d6b35fe961dd5868fe019b7e2cf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/38c4aca9ef3a0301937a6786f0836e7e5e498d6b35fe961dd5868fe019b7e2cf.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libtsan2_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files", "@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libtsan2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d8e04be2cd7f8299668020b1c2a13ce07a1b79e73c901338a6fabd77ccabf004",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d8e04be2cd7f8299668020b1c2a13ce07a1b79e73c901338a6fabd77ccabf004.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libtsan2_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files", "@debian12_libgcc-s1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libubsan1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "145b39c5fa7e70daea98da0e1f66c133339c8e30f45a6147a1201297a5eac29b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/145b39c5fa7e70daea98da0e1f66c133339c8e30f45a6147a1201297a5eac29b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libubsan1_12.2.0-14_arm64.deb"],
        deps = ["@debian12_gcc-12-base_aarch64//:all_files", "@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libubsan1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e46fbb519b4342c114b2fa19bcdb736e294eadc769fae75d6bc2e94a4db67f15",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e46fbb519b4342c114b2fa19bcdb736e294eadc769fae75d6bc2e94a4db67f15.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/g/gcc-12/libubsan1_12.2.0-14_amd64.deb"],
        deps = ["@debian12_gcc-12-base_x86_64//:all_files", "@debian12_libgcc-s1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libudev1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "978b0540c0af58fb55ace7cfcd49bcc9cc5c3f306347440bbd7efe2c3704c3a0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/978b0540c0af58fb55ace7cfcd49bcc9cc5c3f306347440bbd7efe2c3704c3a0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/systemd/libudev1_252.19-1~deb12u1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libudev1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8011853dcb09cd62d60fd95791eabba86df58d70b054f654f1bb51261b95cb98",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8011853dcb09cd62d60fd95791eabba86df58d70b054f654f1bb51261b95cb98.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/systemd/libudev1_252.19-1~deb12u1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libunistring2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "05b0b7700bfe269ff7af61f45e92055d7ef4c532c9584e4e2a352cf0bd4de5b1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/05b0b7700bfe269ff7af61f45e92055d7ef4c532c9584e4e2a352cf0bd4de5b1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libu/libunistring/libunistring2_1.0-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libunistring2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d466bbfe011d764d793c1d9d777cad9c7cf65b938e11598f27408171ad95a951",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d466bbfe011d764d793c1d9d777cad9c7cf65b938e11598f27408171ad95a951.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libu/libunistring/libunistring2_1.0-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libunwind-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "749803d882c86a63b36c5fce83b37ca10deb99ec010e3c67131d7496c3459424",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/749803d882c86a63b36c5fce83b37ca10deb99ec010e3c67131d7496c3459424.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libu/libunwind/libunwind-dev_1.6.2-3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libunwind-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "aa26f38ebf6699c1cb6ee57fd0e42d3a403b37e07bb833e1a4061275484d5b2f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/aa26f38ebf6699c1cb6ee57fd0e42d3a403b37e07bb833e1a4061275484d5b2f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libu/libunwind/libunwind-dev_1.6.2-3_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libunwind8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f83ce7c58caaf15cfb0b6ced538751ea11ded920443aecfe617049ae184b715e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f83ce7c58caaf15cfb0b6ced538751ea11ded920443aecfe617049ae184b715e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libu/libunwind/libunwind8_1.6.2-3_arm64.deb"],
        deps = ["@debian12_liblzma5_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libunwind8_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "7b297868682836e4c87be349f17e4a56bc287586e3576503e84a5cb5485ce925",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7b297868682836e4c87be349f17e4a56bc287586e3576503e84a5cb5485ce925.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libu/libunwind/libunwind8_1.6.2-3_amd64.deb"],
        deps = ["@debian12_liblzma5_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libuuid1_aarch64",
        sha256 = "34d56dc6cbac8a0a4c0db78de67633548268c536b38198f67e6600a7775d3f88",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/34d56dc6cbac8a0a4c0db78de67633548268c536b38198f67e6600a7775d3f88.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/u/util-linux/libuuid1_2.38.1-5+b1_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libuuid1_x86_64",
        sha256 = "c2b3ccade855de14c6ece893a0d2bec63b0a007cbc2970af8152cf06699ccd2a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c2b3ccade855de14c6ece893a0d2bec63b0a007cbc2970af8152cf06699ccd2a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/u/util-linux/libuuid1_2.38.1-5+b1_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b19b4b761a3291da54d7687287359a7147964246ade283939fc8a6e76fb6dc79",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b19b4b761a3291da54d7687287359a7147964246ade283939fc8a6e76fb6dc79.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-dev_2.17.0-1_arm64.deb"],
        deps = ["@debian12_libset-scalar-perl_aarch64//:all_files", "@debian12_libva-drm2_aarch64//:all_files", "@debian12_libva-glx2_aarch64//:all_files", "@debian12_libva-wayland2_aarch64//:all_files", "@debian12_libva-x11-2_aarch64//:all_files", "@debian12_libva2_aarch64//:all_files", "@debian12_libwayland-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ea3ecdc7f01eb4519701ad17616dbd3e1afb28745d61f11f5af4d26bf6f7db4d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ea3ecdc7f01eb4519701ad17616dbd3e1afb28745d61f11f5af4d26bf6f7db4d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-dev_2.17.0-1_amd64.deb"],
        deps = ["@debian12_libset-scalar-perl_x86_64//:all_files", "@debian12_libva-drm2_x86_64//:all_files", "@debian12_libva-glx2_x86_64//:all_files", "@debian12_libva-wayland2_x86_64//:all_files", "@debian12_libva-x11-2_x86_64//:all_files", "@debian12_libva2_x86_64//:all_files", "@debian12_libwayland-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-drm2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "dd1c3981a81171f3886f14b56e2c13cb879111109e5b1e3e786c26641ab2cc2f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/dd1c3981a81171f3886f14b56e2c13cb879111109e5b1e3e786c26641ab2cc2f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-drm2_2.17.0-1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files", "@debian12_libva2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-drm2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "dc0a0cadbdc001df3d89855d8054aacb332a731f98c7741ce580492b0e01e605",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/dc0a0cadbdc001df3d89855d8054aacb332a731f98c7741ce580492b0e01e605.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-drm2_2.17.0-1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files", "@debian12_libva2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-glx2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9f93326415f364e23241030f2bbe6c8a4dc7665cd53f4e06d6b188eb2627fdd6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9f93326415f364e23241030f2bbe6c8a4dc7665cd53f4e06d6b188eb2627fdd6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-glx2_2.17.0-1_arm64.deb"],
        deps = ["@debian12_libgl1_aarch64//:all_files", "@debian12_libva-x11-2_aarch64//:all_files", "@debian12_libva2_aarch64//:all_files", "@debian12_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-glx2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "881f1ef2343c975d06fd1ca7a99fe46790efe263804bb343a9bf16fed0361e4c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/881f1ef2343c975d06fd1ca7a99fe46790efe263804bb343a9bf16fed0361e4c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-glx2_2.17.0-1_amd64.deb"],
        deps = ["@debian12_libgl1_x86_64//:all_files", "@debian12_libva-x11-2_x86_64//:all_files", "@debian12_libva2_x86_64//:all_files", "@debian12_libx11-6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-wayland2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f314ae3f7bc4c58f8012cf0e9d6f5cfa4948a683e0e3243ff18f9e289cb99b72",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f314ae3f7bc4c58f8012cf0e9d6f5cfa4948a683e0e3243ff18f9e289cb99b72.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-wayland2_2.17.0-1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files", "@debian12_libva2_aarch64//:all_files", "@debian12_libwayland-client0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-wayland2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d3334ddbfac5ac006bc7d8fd4f77a3b80a73181895205433b93b537e704d2ff1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d3334ddbfac5ac006bc7d8fd4f77a3b80a73181895205433b93b537e704d2ff1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-wayland2_2.17.0-1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files", "@debian12_libva2_x86_64//:all_files", "@debian12_libwayland-client0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-x11-2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "326f1a6d764adae4ccdc606913ba60b3aecc62788c509444751e7fea75a13584",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/326f1a6d764adae4ccdc606913ba60b3aecc62788c509444751e7fea75a13584.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-x11-2_2.17.0-1_arm64.deb"],
        deps = ["@debian12_libdrm2_aarch64//:all_files", "@debian12_libva2_aarch64//:all_files", "@debian12_libx11-6_aarch64//:all_files", "@debian12_libx11-xcb1_aarch64//:all_files", "@debian12_libxcb-dri3-0_aarch64//:all_files", "@debian12_libxcb1_aarch64//:all_files", "@debian12_libxext6_aarch64//:all_files", "@debian12_libxfixes3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva-x11-2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "95db1ecd8c2d1c3f99a750f1b2d9ba1959b7cd9cbd7eabfc1b8a86a095b379a6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/95db1ecd8c2d1c3f99a750f1b2d9ba1959b7cd9cbd7eabfc1b8a86a095b379a6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva-x11-2_2.17.0-1_amd64.deb"],
        deps = ["@debian12_libdrm2_x86_64//:all_files", "@debian12_libva2_x86_64//:all_files", "@debian12_libx11-6_x86_64//:all_files", "@debian12_libx11-xcb1_x86_64//:all_files", "@debian12_libxcb-dri3-0_x86_64//:all_files", "@debian12_libxcb1_x86_64//:all_files", "@debian12_libxext6_x86_64//:all_files", "@debian12_libxfixes3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b737136a783067f8f202115ed820d955aea16b95a6f06c406cc4305f99d20d3e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b737136a783067f8f202115ed820d955aea16b95a6f06c406cc4305f99d20d3e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva2_2.17.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libva2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e6acf0c4071c01df7c35d18a97768047294dc43d252caae9ccddc1e507456695",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e6acf0c4071c01df7c35d18a97768047294dc43d252caae9ccddc1e507456695.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libva/libva2_2.17.0-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libvdpau1_aarch64",
        sha256 = "2f13f0bce02b7c5c5caecf5522e22852e296d93b2275c6a445dbf5427605b93c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2f13f0bce02b7c5c5caecf5522e22852e296d93b2275c6a445dbf5427605b93c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libvdpau/libvdpau1_1.5-2_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libx11-6_aarch64//:all_files", "@debian12_libxext6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libvdpau1_x86_64",
        sha256 = "517039c1bfbfc453c4b952a068a1b02b77bf46bbf7af882ad877a581bf2c0e21",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/517039c1bfbfc453c4b952a068a1b02b77bf46bbf7af882ad877a581bf2c0e21.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libvdpau/libvdpau1_1.5-2_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libx11-6_x86_64//:all_files", "@debian12_libxext6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libvpx-dev_aarch64",
        sha256 = "921630c0e4d44073dbe930e59c944c5d2a0ce58586c94ab38b87ac9b2697dd6d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/921630c0e4d44073dbe930e59c944c5d2a0ce58586c94ab38b87ac9b2697dd6d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libvpx/libvpx-dev_1.12.0-1+deb12u2_arm64.deb"],
        deps = ["@debian12_libvpx7_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libvpx-dev_x86_64",
        sha256 = "5eaf2e9ec92e689c524322b9aaf7c3b8f23d2f863c03a09646abe5c07bde5c96",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5eaf2e9ec92e689c524322b9aaf7c3b8f23d2f863c03a09646abe5c07bde5c96.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libvpx/libvpx-dev_1.12.0-1+deb12u2_amd64.deb"],
        deps = ["@debian12_libvpx7_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libvpx7_aarch64",
        sha256 = "99c0f4ecaf5308f2c00b9634ecafd8b833228dbda3f4717961b583d3a815361d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/99c0f4ecaf5308f2c00b9634ecafd8b833228dbda3f4717961b583d3a815361d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libvpx/libvpx7_1.12.0-1+deb12u2_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libvpx7_x86_64",
        sha256 = "5ccc3485f9cd2eafa32d163e4368a96722fe18440fb28906e3073c31ce742d60",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5ccc3485f9cd2eafa32d163e4368a96722fe18440fb28906e3073c31ce742d60.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libv/libvpx/libvpx7_1.12.0-1+deb12u2_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "02e9361617b43b130ff032873a910258884160c169e89a1ca45d34d1150a2615",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/02e9361617b43b130ff032873a910258884160c169e89a1ca45d34d1150a2615.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-bin_1.21.0-1_arm64.deb"],
        deps = ["@debian12_libexpat1_aarch64//:all_files", "@debian12_libxml2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-bin_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c07fcba0849b8ca4ed744bbb820fa08a82213f8a46261024005a4d5bbc47afc1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c07fcba0849b8ca4ed744bbb820fa08a82213f8a46261024005a4d5bbc47afc1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-bin_1.21.0-1_amd64.deb"],
        deps = ["@debian12_libexpat1_x86_64//:all_files", "@debian12_libxml2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-client0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ae390cc04c2eb1de90b9a6373505b22c730ada5e72daa50c507b7f99c12faf06",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ae390cc04c2eb1de90b9a6373505b22c730ada5e72daa50c507b7f99c12faf06.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-client0_1.21.0-1_arm64.deb"],
        deps = ["@debian12_libffi8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-client0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1f002d028b8b79eec9847c636d8886d10dfe8c884cc2bebe18086b1391c5a28d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1f002d028b8b79eec9847c636d8886d10dfe8c884cc2bebe18086b1391c5a28d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-client0_1.21.0-1_amd64.deb"],
        deps = ["@debian12_libffi8_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-cursor0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8c8d1a3942bc3fc9dd3ea7c679c1a314e59cdaaa700883d54d1bb3ec89db256b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8c8d1a3942bc3fc9dd3ea7c679c1a314e59cdaaa700883d54d1bb3ec89db256b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-cursor0_1.21.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-cursor0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "0b684467ec795024fcaae99ea4384959cc4052b2da5adec38c9b76dfed9bcf41",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0b684467ec795024fcaae99ea4384959cc4052b2da5adec38c9b76dfed9bcf41.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-cursor0_1.21.0-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "932405c58195d299495230792e374ad304adadf3d3ffdc07eda3466b756adc33",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/932405c58195d299495230792e374ad304adadf3d3ffdc07eda3466b756adc33.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-dev_1.21.0-1_arm64.deb"],
        deps = ["@debian12_libffi-dev_aarch64//:all_files", "@debian12_libwayland-bin_aarch64//:all_files", "@debian12_libwayland-client0_aarch64//:all_files", "@debian12_libwayland-cursor0_aarch64//:all_files", "@debian12_libwayland-egl1_aarch64//:all_files", "@debian12_libwayland-server0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "063ac40e71069a1a92c2be32799f4a2d234f3d361ba98cea2e3137a7ab6804d1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/063ac40e71069a1a92c2be32799f4a2d234f3d361ba98cea2e3137a7ab6804d1.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-dev_1.21.0-1_amd64.deb"],
        deps = ["@debian12_libffi-dev_x86_64//:all_files", "@debian12_libwayland-bin_x86_64//:all_files", "@debian12_libwayland-client0_x86_64//:all_files", "@debian12_libwayland-cursor0_x86_64//:all_files", "@debian12_libwayland-egl1_x86_64//:all_files", "@debian12_libwayland-server0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-egl1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c08124a4d9af24f058b45c88867a84697882e755ea94965795d152fe26fe57c9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c08124a4d9af24f058b45c88867a84697882e755ea94965795d152fe26fe57c9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-egl1_1.21.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-egl1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "cb4225f0e15e00a3570d4cbc288c87ff477d73e741c3f1c5a6d47cf350d45378",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cb4225f0e15e00a3570d4cbc288c87ff477d73e741c3f1c5a6d47cf350d45378.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-egl1_1.21.0-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-server0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4d99e7e0503fdaa999e67d47d8612c0465e775eafa47819573da50bdb09064f2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4d99e7e0503fdaa999e67d47d8612c0465e775eafa47819573da50bdb09064f2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-server0_1.21.0-1_arm64.deb"],
        deps = ["@debian12_libffi8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libwayland-server0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "df0396221d7b794496a687ec61fae82b6465648bc0ab6501ba0a5ed7f56eb8d6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/df0396221d7b794496a687ec61fae82b6465648bc0ab6501ba0a5ed7f56eb8d6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/w/wayland/libwayland-server0_1.21.0-1_amd64.deb"],
        deps = ["@debian12_libffi8_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libx11-6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d1d533e983582282a9ea82c87ac5ce715a9b67bd6d1acbd2439a11c63c36549b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d1d533e983582282a9ea82c87ac5ce715a9b67bd6d1acbd2439a11c63c36549b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libx11/libx11-6_1.8.4-2+deb12u2_arm64.deb"],
        deps = ["@debian12_libx11-data_aarch64//:all_files", "@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libx11-6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d88c973e79fd9b65838d77624142952757e47a6eb1a58602acf0911cf35989f4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d88c973e79fd9b65838d77624142952757e47a6eb1a58602acf0911cf35989f4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libx11/libx11-6_1.8.4-2+deb12u2_amd64.deb"],
        deps = ["@debian12_libx11-data_x86_64//:all_files", "@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libx11-data_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "987a848aeb1c358e4186368871b0526f10bb14c6b53214ab3bf8b69abb830191",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/987a848aeb1c358e4186368871b0526f10bb14c6b53214ab3bf8b69abb830191.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libx11/libx11-data_1.8.4-2+deb12u2_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libx11-data_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "987a848aeb1c358e4186368871b0526f10bb14c6b53214ab3bf8b69abb830191",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/987a848aeb1c358e4186368871b0526f10bb14c6b53214ab3bf8b69abb830191.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libx11/libx11-data_1.8.4-2+deb12u2_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libx11-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "34bfc833561de983cfb4475523fedc13abb4faee14a876300de659591779add8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/34bfc833561de983cfb4475523fedc13abb4faee14a876300de659591779add8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libx11/libx11-dev_1.8.4-2+deb12u2_arm64.deb"],
        deps = ["@debian12_libx11-6_aarch64//:all_files", "@debian12_libxau-dev_aarch64//:all_files", "@debian12_libxcb1-dev_aarch64//:all_files", "@debian12_libxdmcp-dev_aarch64//:all_files", "@debian12_x11proto-dev_aarch64//:all_files", "@debian12_xtrans-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libx11-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8493220d4309af1907a1f2f6eeb204c8103dafcc368394fbc4a0858c28612ff9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8493220d4309af1907a1f2f6eeb204c8103dafcc368394fbc4a0858c28612ff9.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libx11/libx11-dev_1.8.4-2+deb12u2_amd64.deb"],
        deps = ["@debian12_libx11-6_x86_64//:all_files", "@debian12_libxau-dev_x86_64//:all_files", "@debian12_libxcb1-dev_x86_64//:all_files", "@debian12_libxdmcp-dev_x86_64//:all_files", "@debian12_x11proto-dev_x86_64//:all_files", "@debian12_xtrans-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libx11-xcb1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "21c7e6f19909bb1e02bb3c2089789791858c54d37e86cef9d7872782c1cb711e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/21c7e6f19909bb1e02bb3c2089789791858c54d37e86cef9d7872782c1cb711e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libx11/libx11-xcb1_1.8.4-2+deb12u2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libx11-xcb1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f5da45e1d881a793250a96613f28c471a248877f1a0f18a5c90e2a620a76c898",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f5da45e1d881a793250a96613f28c471a248877f1a0f18a5c90e2a620a76c898.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libx11/libx11-xcb1_1.8.4-2+deb12u2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxau-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "21db3761a8f388fc83707c00fac645d4332737a859dd727571e1e6ede003b048",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/21db3761a8f388fc83707c00fac645d4332737a859dd727571e1e6ede003b048.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxau/libxau-dev_1.0.9-1_arm64.deb"],
        deps = ["@debian12_libxau6_aarch64//:all_files", "@debian12_x11proto-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxau-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d1a7f5d484e0879b3b2e8d512894744505e53d078712ce65903fef2ecfd824bb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d1a7f5d484e0879b3b2e8d512894744505e53d078712ce65903fef2ecfd824bb.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxau/libxau-dev_1.0.9-1_amd64.deb"],
        deps = ["@debian12_libxau6_x86_64//:all_files", "@debian12_x11proto-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxau6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "36c2bf400641a80521093771dc2562c903df4065f9eb03add50d90564337ea6c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/36c2bf400641a80521093771dc2562c903df4065f9eb03add50d90564337ea6c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxau/libxau6_1.0.9-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxau6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "679db1c4579ec7c61079adeaae8528adeb2e4bf5465baa6c56233b995d714750",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/679db1c4579ec7c61079adeaae8528adeb2e4bf5465baa6c56233b995d714750.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxau/libxau6_1.0.9-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-dri2-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ec012de2f5945a5662b32ab1eb59be944e4b5b5ddae89ae28765819df62817f6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ec012de2f5945a5662b32ab1eb59be944e4b5b5ddae89ae28765819df62817f6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-dri2-0_1.15-1_arm64.deb"],
        deps = ["@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-dri2-0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ef4959aa9e09a0d38d1de432e747585129d5d2dc1d84c8b6b3d2ffc3708b5805",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ef4959aa9e09a0d38d1de432e747585129d5d2dc1d84c8b6b3d2ffc3708b5805.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-dri2-0_1.15-1_amd64.deb"],
        deps = ["@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-dri3-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "a4b2a700d0da1dcaddc6435986d2ae45e8efdb58a2892a2e87cebb0c85dca359",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a4b2a700d0da1dcaddc6435986d2ae45e8efdb58a2892a2e87cebb0c85dca359.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-dri3-0_1.15-1_arm64.deb"],
        deps = ["@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-dri3-0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "02699b144b9467de8636d27a76984b8f4e7b66e2d25d96df2b9677be86ee9a29",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/02699b144b9467de8636d27a76984b8f4e7b66e2d25d96df2b9677be86ee9a29.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-dri3-0_1.15-1_amd64.deb"],
        deps = ["@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-glx0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "4ad61651315a9267e4447af9cc1a63b08eded03697af34fa65cc3128f5466e37",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4ad61651315a9267e4447af9cc1a63b08eded03697af34fa65cc3128f5466e37.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-glx0_1.15-1_arm64.deb"],
        deps = ["@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-glx0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1bce55fc292d93fa5f7fa50f84cef99ec29be70d0ffe98e86b8008e59f4a34fa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1bce55fc292d93fa5f7fa50f84cef99ec29be70d0ffe98e86b8008e59f4a34fa.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-glx0_1.15-1_amd64.deb"],
        deps = ["@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-present0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "df2148b69f25b6a31af73bcfb355b2ae3724678167457996febb3d335377ca17",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/df2148b69f25b6a31af73bcfb355b2ae3724678167457996febb3d335377ca17.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-present0_1.15-1_arm64.deb"],
        deps = ["@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-present0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "89383e627a4d17b9390d609b2459481bfd2029566367b43068586769e418b6e5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/89383e627a4d17b9390d609b2459481bfd2029566367b43068586769e418b6e5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-present0_1.15-1_amd64.deb"],
        deps = ["@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-randr0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9e09fe9b7f6ac596f8b14499e9a10c336b7d3ff27e78f751dffeac9975113004",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9e09fe9b7f6ac596f8b14499e9a10c336b7d3ff27e78f751dffeac9975113004.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-randr0_1.15-1_arm64.deb"],
        deps = ["@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-randr0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f86e3d8ff8622871008833e9d064919b7a6237399c903c59fc330ff00f199ff5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f86e3d8ff8622871008833e9d064919b7a6237399c903c59fc330ff00f199ff5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-randr0_1.15-1_amd64.deb"],
        deps = ["@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-shm0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "825931950d42f51cd795cf2e5b9399d840225491e95ec530e0c5a9145c4ba5a0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/825931950d42f51cd795cf2e5b9399d840225491e95ec530e0c5a9145c4ba5a0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-shm0_1.15-1_arm64.deb"],
        deps = ["@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-shm0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c1afcef29dc78b95c475159b181b28b1dedaf1d5aa06efd2fa6d90c73bfbe0e5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c1afcef29dc78b95c475159b181b28b1dedaf1d5aa06efd2fa6d90c73bfbe0e5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-shm0_1.15-1_amd64.deb"],
        deps = ["@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-sync1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "63f165fb851ffacf021cdc4224fb93f5694a5c03dc699f2f4237aa67e1550d25",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/63f165fb851ffacf021cdc4224fb93f5694a5c03dc699f2f4237aa67e1550d25.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-sync1_1.15-1_arm64.deb"],
        deps = ["@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-sync1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3366ce715220d38dd0148b78a8e738137bade25ef7eec0698850c6f66800844f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3366ce715220d38dd0148b78a8e738137bade25ef7eec0698850c6f66800844f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-sync1_1.15-1_amd64.deb"],
        deps = ["@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-xfixes0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e6cfc1f2a68d4c7f0c10e8bf24ebab20d791ba96ec1ca70824ccd927f0fa9e21",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e6cfc1f2a68d4c7f0c10e8bf24ebab20d791ba96ec1ca70824ccd927f0fa9e21.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-xfixes0_1.15-1_arm64.deb"],
        deps = ["@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb-xfixes0_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d744a7ebad2cbcf301c96cd6a1ab3ee856e436fc7be5cff5b7c28ac2ac181a64",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d744a7ebad2cbcf301c96cd6a1ab3ee856e436fc7be5cff5b7c28ac2ac181a64.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb-xfixes0_1.15-1_amd64.deb"],
        deps = ["@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb1-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "87b5c3f1bb130727df68295ecfaadc305ee10c150afeb1a183e95b9150e18db5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/87b5c3f1bb130727df68295ecfaadc305ee10c150afeb1a183e95b9150e18db5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb1-dev_1.15-1_arm64.deb"],
        deps = ["@debian12_libpthread-stubs0-dev_aarch64//:all_files", "@debian12_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb1-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c078c024114fdada06d3158af1771d7ed8763ab434cfbcbe6a334aa8a9cae358",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c078c024114fdada06d3158af1771d7ed8763ab434cfbcbe6a334aa8a9cae358.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb1-dev_1.15-1_amd64.deb"],
        deps = ["@debian12_libpthread-stubs0-dev_x86_64//:all_files", "@debian12_libxcb1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "041d9a68415c3ccf3ce8f4f8b88e4bbfb5dc1f0d97013c6ef8423e620ea50f84",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/041d9a68415c3ccf3ce8f4f8b88e4bbfb5dc1f0d97013c6ef8423e620ea50f84.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb1_1.15-1_arm64.deb"],
        deps = ["@debian12_libxau6_aarch64//:all_files", "@debian12_libxdmcp6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxcb1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fdc61332a3892168f3cc9cfa1fe9cf11a91dc3e0acacbc47cbc50ebaa234cc71",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fdc61332a3892168f3cc9cfa1fe9cf11a91dc3e0acacbc47cbc50ebaa234cc71.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxcb/libxcb1_1.15-1_amd64.deb"],
        deps = ["@debian12_libxau6_x86_64//:all_files", "@debian12_libxdmcp6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxdmcp-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6fd0bc51f3a4e6b7f944f24f15524a4c4ba68a4b0aa136c18bdb96a2d36988f2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6fd0bc51f3a4e6b7f944f24f15524a4c4ba68a4b0aa136c18bdb96a2d36988f2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxdmcp/libxdmcp-dev_1.1.2-3_arm64.deb"],
        deps = ["@debian12_libxdmcp6_aarch64//:all_files", "@debian12_x11proto-core-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxdmcp-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c6733e5f6463afd261998e408be6eb37f24ce0a64b63bed50a87ddb18ebc1699",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c6733e5f6463afd261998e408be6eb37f24ce0a64b63bed50a87ddb18ebc1699.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxdmcp/libxdmcp-dev_1.1.2-3_amd64.deb"],
        deps = ["@debian12_libxdmcp6_x86_64//:all_files", "@debian12_x11proto-core-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxdmcp6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e92569ac33247261aa09adfadc34ced3994ac301cf8b58de415a2d5dbf15ccfc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e92569ac33247261aa09adfadc34ced3994ac301cf8b58de415a2d5dbf15ccfc.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxdmcp/libxdmcp6_1.1.2-3_arm64.deb"],
        deps = ["@debian12_libbsd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxdmcp6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ecb8536f5fb34543b55bb9dc5f5b14c9dbb4150a7bddb3f2287b7cab6e9d25ef",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ecb8536f5fb34543b55bb9dc5f5b14c9dbb4150a7bddb3f2287b7cab6e9d25ef.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxdmcp/libxdmcp6_1.1.2-3_amd64.deb"],
        deps = ["@debian12_libbsd0_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxext6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5e9c0ad606eb4674c645fe8e0e64330c47d2729f7a59ed569848610efd5d5b62",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5e9c0ad606eb4674c645fe8e0e64330c47d2729f7a59ed569848610efd5d5b62.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxext/libxext6_1.3.4-1+b1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxext6_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "504b7be9d7df4f6f4519e8dd4d6f9d03a9fb911a78530fa23a692fba3058cba6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/504b7be9d7df4f6f4519e8dd4d6f9d03a9fb911a78530fa23a692fba3058cba6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxext/libxext6_1.3.4-1+b1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxfixes3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d47bda8fed01b19b41d503e2df05d9166c58e30e2376f2f8784ceb7a834befe6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d47bda8fed01b19b41d503e2df05d9166c58e30e2376f2f8784ceb7a834befe6.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxfixes/libxfixes3_6.0.0-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxfixes3_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1cd616396ff2ecae77e6e8b5b7695d414f0146de2d147837a2a02165f99e1a2c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1cd616396ff2ecae77e6e8b5b7695d414f0146de2d147837a2a02165f99e1a2c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxfixes/libxfixes3_6.0.0-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxml2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9147f4b4c0ec4c2a4cbe8f1fd2e38746c28f80fcc59c9febfd2aa0a22c6cbfe0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9147f4b4c0ec4c2a4cbe8f1fd2e38746c28f80fcc59c9febfd2aa0a22c6cbfe0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxml2/libxml2_2.9.14+dfsg-1.3~deb12u1_arm64.deb"],
        deps = ["@debian12_libicu72_aarch64//:all_files", "@debian12_liblzma5_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxml2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "35b76cb7038fc1c940204a4f05f33ffb79d027353ce469397d9adcf8f9b3e1a7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/35b76cb7038fc1c940204a4f05f33ffb79d027353ce469397d9adcf8f9b3e1a7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxml2/libxml2_2.9.14+dfsg-1.3~deb12u1_amd64.deb"],
        deps = ["@debian12_libicu72_x86_64//:all_files", "@debian12_liblzma5_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxshmfence1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "fa2edaae6902681ecd02534dcdf8389ac48714d3385d572cc17747160957acc8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fa2edaae6902681ecd02534dcdf8389ac48714d3385d572cc17747160957acc8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxshmfence/libxshmfence1_1.3-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxshmfence1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "1a38142e40e3d32dc4f9a326bf5617363b7d9b4bb762fdcdd262f2192092024d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1a38142e40e3d32dc4f9a326bf5617363b7d9b4bb762fdcdd262f2192092024d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxshmfence/libxshmfence1_1.3-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxtables12_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8884a42f98b989e5ba2e91d931b5c6e9eb72568e5d3001d97d73d7913890a7b3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8884a42f98b989e5ba2e91d931b5c6e9eb72568e5d3001d97d73d7913890a7b3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/iptables/libxtables12_1.8.9-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxtables12_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "cb841d66950a43af4a398625313d2f3da9065299c9738538de6c2c3495857040",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cb841d66950a43af4a398625313d2f3da9065299c9738538de6c2c3495857040.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/i/iptables/libxtables12_1.8.9-2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxxf86vm1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "8a4826772ac480804d6b8d776a4130d95260c036b9b218de4c8a0f07eb9c5bba",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8a4826772ac480804d6b8d776a4130d95260c036b9b218de4c8a0f07eb9c5bba.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxxf86vm/libxxf86vm1_1.1.4-1+b2_arm64.deb"],
        deps = ["@debian12_libx11-6_aarch64//:all_files", "@debian12_libxext6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libxxf86vm1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6f4ca916aaec26d7000fa7f58de3f71119309ab7590ce1f517abfe1825a676c7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6f4ca916aaec26d7000fa7f58de3f71119309ab7590ce1f517abfe1825a676c7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libx/libxxf86vm/libxxf86vm1_1.1.4-1+b2_amd64.deb"],
        deps = ["@debian12_libx11-6_x86_64//:all_files", "@debian12_libxext6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libyajl2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "97f1216dffee32d3d4d71194f466432dd5bc9c9d9a2eb435fd12cb58dff8582c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/97f1216dffee32d3d4d71194f466432dd5bc9c9d9a2eb435fd12cb58dff8582c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/y/yajl/libyajl2_2.1.0-3+deb12u2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libyajl2_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b512a6efbe735e7c731185a1b3ea477ff306f0c3f6aa2970e85b12f01fe8063b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b512a6efbe735e7c731185a1b3ea477ff306f0c3f6aa2970e85b12f01fe8063b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/y/yajl/libyajl2_2.1.0-3+deb12u2_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libz3-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "115080b0aee2d75f316da4be24c8a5e88ed7c362c7b4a4cd13f3af9b9da130f2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/115080b0aee2d75f316da4be24c8a5e88ed7c362c7b4a4cd13f3af9b9da130f2.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/z/z3/libz3-4_4.8.12-3.1_arm64.deb"],
        deps = ["@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libz3-4_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6221ca25ad5abcfbe1965801029d85a88b4775320384b4b716de8fab7a4d2f7a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6221ca25ad5abcfbe1965801029d85a88b4775320384b4b716de8fab7a4d2f7a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/z/z3/libz3-4_4.8.12-3.1_amd64.deb"],
        deps = ["@debian12_libgcc-s1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libzstd1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "95e173c9538f96ede4fc275ec7863f395a97dd0ea62454be9bc914efa1b9be93",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/95e173c9538f96ede4fc275ec7863f395a97dd0ea62454be9bc914efa1b9be93.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libz/libzstd/libzstd1_1.5.4+dfsg2-5_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_libzstd1_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "6315b5ac38b724a710fb96bf1042019398cb656718b1522279a5185ed39318fa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6315b5ac38b724a710fb96bf1042019398cb656718b1522279a5185ed39318fa.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libz/libzstd/libzstd1_1.5.4+dfsg2-5_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_linux-libc-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "709eb551a8a86d8ac650690faefc74f639556800d8435e760a0ae3d68e90461f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/709eb551a8a86d8ac650690faefc74f639556800d8435e760a0ae3d68e90461f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/linux/linux-libc-dev_6.1.64-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_linux-libc-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "dcc4316d1caa241392f50f13585c0f27eba3392340420bb247181cef6b15c66a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/dcc4316d1caa241392f50f13585c0f27eba3392340420bb247181cef6b15c66a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/linux/linux-libc-dev_6.1.64-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_lsb-base_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f8bedd167280e76636df3a1bc023cd2906d458916c1af4c1d7912c5b971fc642",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f8bedd167280e76636df3a1bc023cd2906d458916c1af4c1d7912c5b971fc642.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lsb/lsb-base_11.6_all.deb"],
        deps = ["@debian12_sysvinit-utils_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_lsb-base_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f8bedd167280e76636df3a1bc023cd2906d458916c1af4c1d7912c5b971fc642",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f8bedd167280e76636df3a1bc023cd2906d458916c1af4c1d7912c5b971fc642.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/l/lsb/lsb-base_11.6_all.deb"],
        deps = ["@debian12_sysvinit-utils_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_media-types_aarch64",
        sha256 = "aaa46dcb3b39948ae2e0fdb72cfcb2f48c0b59f19785a3da8045c05eb19955dd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/aaa46dcb3b39948ae2e0fdb72cfcb2f48c0b59f19785a3da8045c05eb19955dd.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/media-types/media-types_10.0.0_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_media-types_x86_64",
        sha256 = "aaa46dcb3b39948ae2e0fdb72cfcb2f48c0b59f19785a3da8045c05eb19955dd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/aaa46dcb3b39948ae2e0fdb72cfcb2f48c0b59f19785a3da8045c05eb19955dd.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/media-types/media-types_10.0.0_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_mesa-common-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "44ed27677ad2226b62daba25396ff22cf1b880359f5b6adcd40ab98f93f76554",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/44ed27677ad2226b62daba25396ff22cf1b880359f5b6adcd40ab98f93f76554.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/mesa-common-dev_22.3.6-1+deb12u1_arm64.deb"],
        deps = ["@debian12_libdrm-dev_aarch64//:all_files", "@debian12_libgl-dev_aarch64//:all_files", "@debian12_libglx-dev_aarch64//:all_files", "@debian12_libx11-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_mesa-common-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3816dd27b3ef2abc92dfd6282bf018b4331fe7aebb4fbc52daf249e71a865236",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3816dd27b3ef2abc92dfd6282bf018b4331fe7aebb4fbc52daf249e71a865236.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/m/mesa/mesa-common-dev_22.3.6-1+deb12u1_amd64.deb"],
        deps = ["@debian12_libdrm-dev_x86_64//:all_files", "@debian12_libgl-dev_x86_64//:all_files", "@debian12_libglx-dev_x86_64//:all_files", "@debian12_libx11-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_nasm_aarch64",
        sha256 = "291e7bd6c76358bc44f0a5b5dae2320f3d63f6a1da004d70b3f024de5f5fa493",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/291e7bd6c76358bc44f0a5b5dae2320f3d63f6a1da004d70b3f024de5f5fa493.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/nasm/nasm_2.16.01-1_arm64.deb"],
        deps = ["@debian12_dpkg_aarch64//:all_files", "@debian12_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_nasm_x86_64",
        sha256 = "0180693ad5d2da8b36e7d73b9b596dc3e9f900238e3de699976c0576875b878f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0180693ad5d2da8b36e7d73b9b596dc3e9f900238e3de699976c0576875b878f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/nasm/nasm_2.16.01-1_amd64.deb"],
        deps = ["@debian12_dpkg_x86_64//:all_files", "@debian12_libc6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_ncurses-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "59a56e4bfd04716771bcae67465e6bb87e535da0b232b04d30cf1a920a869b40",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/59a56e4bfd04716771bcae67465e6bb87e535da0b232b04d30cf1a920a869b40.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/ncurses-bin_6.4-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_ncurses-bin_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "396d6e453aee6d71b7141f0bfb333a6c08a44c64f77632bdf52894ccd123db46",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/396d6e453aee6d71b7141f0bfb333a6c08a44c64f77632bdf52894ccd123db46.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/ncurses/ncurses-bin_6.4-4_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_netavark_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "35abb684b8bf4780d353b1a02216766d4c218fac48d32eaa3ac80877aff1a3d7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/35abb684b8bf4780d353b1a02216766d4c218fac48d32eaa3ac80877aff1a3d7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/netavark/netavark_1.4.0-3_arm64.deb"],
        deps = ["@debian12_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_netavark_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "e5d6c9efec8b050009cae14b7c740d3d9430e87047003dbad8498324f89516f7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e5d6c9efec8b050009cae14b7c740d3d9430e87047003dbad8498324f89516f7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/netavark/netavark_1.4.0-3_amd64.deb"],
        deps = ["@debian12_libgcc-s1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_netbase_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "29b23c48c0fe6f878e56c5ddc9f65d1c05d729360f3690a593a8c795031cd867",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/29b23c48c0fe6f878e56c5ddc9f65d1c05d729360f3690a593a8c795031cd867.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/netbase/netbase_6.4_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_netbase_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "29b23c48c0fe6f878e56c5ddc9f65d1c05d729360f3690a593a8c795031cd867",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/29b23c48c0fe6f878e56c5ddc9f65d1c05d729360f3690a593a8c795031cd867.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/n/netbase/netbase_6.4_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_openssl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ef8ec413c2bd866dbe7ab068cd45b28d22e6c586e3ab0bd8de127a3e8dfe650b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ef8ec413c2bd866dbe7ab068cd45b28d22e6c586e3ab0bd8de127a3e8dfe650b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/o/openssl/openssl_3.0.11-1~deb12u2_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libssl3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_openssl_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "ea063646d4f70d15be5ed52b67b5ac95d68dda823c60d808c7c25439c6d14e4d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ea063646d4f70d15be5ed52b67b5ac95d68dda823c60d808c7c25439c6d14e4d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/o/openssl/openssl_3.0.11-1~deb12u2_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libssl3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_passwd_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "c8e689ebef5c3ad4fb39ea8b0d49c33a483879dd0f477a07d710f7609809d697",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c8e689ebef5c3ad4fb39ea8b0d49c33a483879dd0f477a07d710f7609809d697.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/shadow/passwd_4.13+dfsg1-1+b1_arm64.deb"],
        deps = ["@debian12_libaudit1_aarch64//:all_files", "@debian12_libcrypt1_aarch64//:all_files", "@debian12_libpam-modules_aarch64//:all_files", "@debian12_libpam0g_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files", "@debian12_libsemanage2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_passwd_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "343b60a755ceb2c3687f9a5c9c9dc00eea0e44a7de49a537c36df17894f784b3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/343b60a755ceb2c3687f9a5c9c9dc00eea0e44a7de49a537c36df17894f784b3.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/shadow/passwd_4.13+dfsg1-1+b1_amd64.deb"],
        deps = ["@debian12_libaudit1_x86_64//:all_files", "@debian12_libcrypt1_x86_64//:all_files", "@debian12_libpam-modules_x86_64//:all_files", "@debian12_libpam0g_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files", "@debian12_libsemanage2_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_perl-base_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "534d5f3fb0664b74d2174040e23c2a3568a81c6766fddbc57677b0d555b0713b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/534d5f3fb0664b74d2174040e23c2a3568a81c6766fddbc57677b0d555b0713b.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/perl/perl-base_5.36.0-7+deb12u1_arm64.deb"],
        deps = ["@debian12_libcrypt1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_perl-base_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "b4327c2d8e2ca92402205ac6b5845b3110fa2a1d50925c0e61c39624583a8baf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b4327c2d8e2ca92402205ac6b5845b3110fa2a1d50925c0e61c39624583a8baf.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/perl/perl-base_5.36.0-7+deb12u1_amd64.deb"],
        deps = ["@debian12_libcrypt1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_perl-modules-5.36_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9b8223674661ead1836ce21966f7e4511a3a943c1b87c02ea92ec17ed2c3f2cf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9b8223674661ead1836ce21966f7e4511a3a943c1b87c02ea92ec17ed2c3f2cf.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/perl/perl-modules-5.36_5.36.0-7+deb12u1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_perl-modules-5.36_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9b8223674661ead1836ce21966f7e4511a3a943c1b87c02ea92ec17ed2c3f2cf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9b8223674661ead1836ce21966f7e4511a3a943c1b87c02ea92ec17ed2c3f2cf.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/perl/perl-modules-5.36_5.36.0-7+deb12u1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_perl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "77ba64ee78ba620dedac03d2ed8171d2af17bc2ea19fda9b572b0b63c86d5d4f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/77ba64ee78ba620dedac03d2ed8171d2af17bc2ea19fda9b572b0b63c86d5d4f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/perl/perl_5.36.0-7+deb12u1_arm64.deb"],
        deps = ["@debian12_libperl5.36_aarch64//:all_files", "@debian12_perl-base_aarch64//:all_files", "@debian12_perl-modules-5.36_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_perl_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "95224197cc1275ee3e625be4522f9d03f8fea3bd7a5d7d8f1f55ab914736b404",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/95224197cc1275ee3e625be4522f9d03f8fea3bd7a5d7d8f1f55ab914736b404.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/perl/perl_5.36.0-7+deb12u1_amd64.deb"],
        deps = ["@debian12_libperl5.36_x86_64//:all_files", "@debian12_perl-base_x86_64//:all_files", "@debian12_perl-modules-5.36_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_pinentry-curses_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "0bdb0e5d908dfcdd3be4fd62c9360bb40bbf4b99601696578c32f61bc99216a4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0bdb0e5d908dfcdd3be4fd62c9360bb40bbf4b99601696578c32f61bc99216a4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pinentry/pinentry-curses_1.2.1-1_arm64.deb"],
        deps = ["@debian12_libassuan0_aarch64//:all_files", "@debian12_libgpg-error0_aarch64//:all_files", "@debian12_libncursesw6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_pinentry-curses_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "27b3d102545f597df9e6dc5c7f6590a648de09b57debd6b05ad3d1189de428d5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/27b3d102545f597df9e6dc5c7f6590a648de09b57debd6b05ad3d1189de428d5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/pinentry/pinentry-curses_1.2.1-1_amd64.deb"],
        deps = ["@debian12_libassuan0_x86_64//:all_files", "@debian12_libgpg-error0_x86_64//:all_files", "@debian12_libncursesw6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_podman_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "30f6521e8be32804a066351b999c760dea391f0da06d5621d58e91ba17476628",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/30f6521e8be32804a066351b999c760dea391f0da06d5621d58e91ba17476628.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libp/libpod/podman_4.3.1+ds1-8+b1_arm64.deb"],
        deps = ["@debian12_conmon_aarch64//:all_files", "@debian12_crun_aarch64//:all_files", "@debian12_golang-github-containers-common_aarch64//:all_files", "@debian12_libdevmapper1.02.1_aarch64//:all_files", "@debian12_libgpgme11_aarch64//:all_files", "@debian12_libseccomp2_aarch64//:all_files", "@debian12_libsubid4_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_podman_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3bff99fdf316ddebef9b60991e6acdd562fad3e17570e107e40be60c8a2a0b37",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3bff99fdf316ddebef9b60991e6acdd562fad3e17570e107e40be60c8a2a0b37.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/libp/libpod/podman_4.3.1+ds1-8+b1_amd64.deb"],
        deps = ["@debian12_conmon_x86_64//:all_files", "@debian12_crun_x86_64//:all_files", "@debian12_golang-github-containers-common_x86_64//:all_files", "@debian12_libdevmapper1.02.1_x86_64//:all_files", "@debian12_libgpgme11_x86_64//:all_files", "@debian12_libseccomp2_x86_64//:all_files", "@debian12_libsubid4_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_python3.11-minimal_aarch64",
        sha256 = "6e83406f07fb2f567aa12d57e51c24c40c42d7f144970ed9a05f0be25a1b1241",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6e83406f07fb2f567aa12d57e51c24c40c42d7f144970ed9a05f0be25a1b1241.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/python3.11/python3.11-minimal_3.11.2-6_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files", "@debian12_libexpat1_aarch64//:all_files", "@debian12_libpython3.11-minimal_aarch64//:all_files", "@debian12_zlib1g_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_python3.11-minimal_x86_64",
        sha256 = "a72247ba64bcd1d0ace2ea8eefd7bcfaca84204def9495269526c25dd9fddc0c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a72247ba64bcd1d0ace2ea8eefd7bcfaca84204def9495269526c25dd9fddc0c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/python3.11/python3.11-minimal_3.11.2-6_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files", "@debian12_libexpat1_x86_64//:all_files", "@debian12_libpython3.11-minimal_x86_64//:all_files", "@debian12_zlib1g_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_python3.11_aarch64",
        extra_symlinks = {
            "usr/bin/python": "/usr/bin/python3",
            "usr/bin/python3": "/usr/bin/python3.11",
        },
        sha256 = "0d2e0b2f1dca43053298c2bd1d279af152f820ffd8c4554d19ecee7b47f2c316",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0d2e0b2f1dca43053298c2bd1d279af152f820ffd8c4554d19ecee7b47f2c316.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/python3.11/python3.11_3.11.2-6_arm64.deb"],
        deps = ["@debian12_libpython3.11-stdlib_aarch64//:all_files", "@debian12_media-types_aarch64//:all_files", "@debian12_python3.11-minimal_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_python3.11_x86_64",
        extra_symlinks = {
            "usr/bin/python": "/usr/bin/python3",
            "usr/bin/python3": "/usr/bin/python3.11",
        },
        sha256 = "9d1d4ba9ac38a7ae48567bfbd0bec88e02a5ccd941a48a76709a131197ea6570",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9d1d4ba9ac38a7ae48567bfbd0bec88e02a5ccd941a48a76709a131197ea6570.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/p/python3.11/python3.11_3.11.2-6_amd64.deb"],
        deps = ["@debian12_libpython3.11-stdlib_x86_64//:all_files", "@debian12_media-types_x86_64//:all_files", "@debian12_python3.11-minimal_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_readline-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "69317523fe56429aa361545416ad339d138c1500e5a604856a80dd9074b4e35c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/69317523fe56429aa361545416ad339d138c1500e5a604856a80dd9074b4e35c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/r/readline/readline-common_8.2-1.3_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_readline-common_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "69317523fe56429aa361545416ad339d138c1500e5a604856a80dd9074b4e35c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/69317523fe56429aa361545416ad339d138c1500e5a604856a80dd9074b4e35c.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/r/readline/readline-common_8.2-1.3_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_sed_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "daf0ece735fc64cf37afce9f987c59cbb1011f2a774ff4b2e1d9b34b0726631d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/daf0ece735fc64cf37afce9f987c59cbb1011f2a774ff4b2e1d9b34b0726631d.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/sed/sed_4.9-1_arm64.deb"],
        deps = ["@debian12_libacl1_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_sed_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "177cacdfe9508448d84bf25534a87a7fcc058d8e2dcd422672851ea13f2115df",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/177cacdfe9508448d84bf25534a87a7fcc058d8e2dcd422672851ea13f2115df.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/sed/sed_4.9-1_amd64.deb"],
        deps = ["@debian12_libacl1_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_sysvinit-utils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "cb75b312466d8b208444f99c9743e1f9846d7f9debccabccdda65123c611c2db",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cb75b312466d8b208444f99c9743e1f9846d7f9debccabccdda65123c611c2db.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/sysvinit/sysvinit-utils_3.06-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_sysvinit-utils_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "11790842108768ec52432ea22e7b4f057232813b7c27ef6dfe1aba776a5cb90e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/11790842108768ec52432ea22e7b4f057232813b7c27ef6dfe1aba776a5cb90e.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/s/sysvinit/sysvinit-utils_3.06-4_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_tar_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        extra_symlinks = {
            "usr/sbin/rmt": "/usr/sbin/rmt-tar",
        },
        sha256 = "c55a4a377cc4fa52b71d6ce5809a5d14f295ce4feed2e689c02be80725f887c8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c55a4a377cc4fa52b71d6ce5809a5d14f295ce4feed2e689c02be80725f887c8.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/t/tar/tar_1.34+dfsg-1.2_arm64.deb"],
        deps = ["@debian12_libacl1_aarch64//:all_files", "@debian12_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_tar_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        extra_symlinks = {
            "usr/sbin/rmt": "/usr/sbin/rmt-tar",
        },
        sha256 = "36a29db2aa4262bd02c23df42cd91cc709883fe52a517aa8a1b148039305eef0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/36a29db2aa4262bd02c23df42cd91cc709883fe52a517aa8a1b148039305eef0.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/t/tar/tar_1.34+dfsg-1.2_amd64.deb"],
        deps = ["@debian12_libacl1_x86_64//:all_files", "@debian12_libselinux1_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_usrmerge_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "069d1a9ceb22ce73bc851d72c93fb580c05bbbd59cd3041f72ea01f17aac4e86",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/069d1a9ceb22ce73bc851d72c93fb580c05bbbd59cd3041f72ea01f17aac4e86.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/u/usrmerge/usrmerge_35_all.deb"],
        deps = ["@debian12_libfile-find-rule-perl_aarch64//:all_files", "@debian12_perl_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_usrmerge_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "069d1a9ceb22ce73bc851d72c93fb580c05bbbd59cd3041f72ea01f17aac4e86",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/069d1a9ceb22ce73bc851d72c93fb580c05bbbd59cd3041f72ea01f17aac4e86.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/u/usrmerge/usrmerge_35_all.deb"],
        deps = ["@debian12_libfile-find-rule-perl_x86_64//:all_files", "@debian12_perl_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_x11proto-core-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5fb73f674e174e766f489039f4e321bd6062986a8306643d71a07e4cc2e4edf7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5fb73f674e174e766f489039f4e321bd6062986a8306643d71a07e4cc2e4edf7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xorgproto/x11proto-core-dev_2022.1-1_all.deb"],
        deps = ["@debian12_x11proto-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_x11proto-core-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "5fb73f674e174e766f489039f4e321bd6062986a8306643d71a07e4cc2e4edf7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5fb73f674e174e766f489039f4e321bd6062986a8306643d71a07e4cc2e4edf7.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xorgproto/x11proto-core-dev_2022.1-1_all.deb"],
        deps = ["@debian12_x11proto-dev_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_x11proto-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3cab66a6591774188a568f9b54c4e7956f18da76e0e0fc4a779db5f6bcc3f148",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3cab66a6591774188a568f9b54c4e7956f18da76e0e0fc4a779db5f6bcc3f148.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xorgproto/x11proto-dev_2022.1-1_all.deb"],
        deps = ["@debian12_xorg-sgml-doctools_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_x11proto-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "3cab66a6591774188a568f9b54c4e7956f18da76e0e0fc4a779db5f6bcc3f148",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3cab66a6591774188a568f9b54c4e7956f18da76e0e0fc4a779db5f6bcc3f148.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xorgproto/x11proto-dev_2022.1-1_all.deb"],
        deps = ["@debian12_xorg-sgml-doctools_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_xorg-sgml-doctools_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "168345058319094e475a87ace66f5fb6ae802109650ea8434d672117982b5d0a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/168345058319094e475a87ace66f5fb6ae802109650ea8434d672117982b5d0a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xorg-sgml-doctools/xorg-sgml-doctools_1.11-1.1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_xorg-sgml-doctools_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "168345058319094e475a87ace66f5fb6ae802109650ea8434d672117982b5d0a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/168345058319094e475a87ace66f5fb6ae802109650ea8434d672117982b5d0a.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xorg-sgml-doctools/xorg-sgml-doctools_1.11-1.1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_xtrans-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9ce1af9464faee0c679348dd11cdf63934c12e734a64e0903692b0cb5af38e06",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9ce1af9464faee0c679348dd11cdf63934c12e734a64e0903692b0cb5af38e06.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xtrans/xtrans-dev_1.4.0-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_xtrans-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "9ce1af9464faee0c679348dd11cdf63934c12e734a64e0903692b0cb5af38e06",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9ce1af9464faee0c679348dd11cdf63934c12e734a64e0903692b0cb5af38e06.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/x/xtrans/xtrans-dev_1.4.0-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_yasm_aarch64",
        sha256 = "16c8569c99da33b0cce63029055f1da255d2d8c5b86a35c27341ae6d6da35a8f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/16c8569c99da33b0cce63029055f1da255d2d8c5b86a35c27341ae6d6da35a8f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/y/yasm/yasm_1.3.0-4_arm64.deb"],
        deps = ["@debian12_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_yasm_x86_64",
        sha256 = "1e2b29b9ad5f469aed7f17f7d23fc34860c469186b4b5274f1759988936e4faa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1e2b29b9ad5f469aed7f17f7d23fc34860c469186b4b5274f1759988936e4faa.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/y/yasm/yasm_1.3.0-4_amd64.deb"],
        deps = ["@debian12_libc6_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_zlib1g-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "afb60f0bae432f2f0edfe2d674e99c1a13a147c9aa2b52f2f23e3df19ece76f5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/afb60f0bae432f2f0edfe2d674e99c1a13a147c9aa2b52f2f23e3df19ece76f5.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/z/zlib/zlib1g-dev_1.2.13.dfsg-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_zlib1g-dev_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "f9ce531f60cbd5df37996af9370e0171be96902a17ec2bdbd8d62038c354094f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f9ce531f60cbd5df37996af9370e0171be96902a17ec2bdbd8d62038c354094f.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/z/zlib/zlib1g-dev_1.2.13.dfsg-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_zlib1g_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "52b8b8a145bbe1956bba82034f77022cbef0c3d0885c9e32d9817a7932fe1913",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/52b8b8a145bbe1956bba82034f77022cbef0c3d0885c9e32d9817a7932fe1913.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/z/zlib/zlib1g_1.2.13.dfsg-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_zlib1g_x86_64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt"],
        sha256 = "d7dd1d1411fedf27f5e27650a6eff20ef294077b568f4c8c5e51466dc7c08ce4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d7dd1d1411fedf27f5e27650a6eff20ef294077b568f4c8c5e51466dc7c08ce4.deb", "https://snapshot.debian.org/archive/debian/20231209T111918Z/pool/main/z/zlib/zlib1g_1.2.13.dfsg-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "intel-compute-runtime_igc-core_x86_64",
        sha256 = "0a3497d36d080f7be822eb10ce2611420543444328d7034ba97ebafc637c77a6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0a3497d36d080f7be822eb10ce2611420543444328d7034ba97ebafc637c77a6.deb", "https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-core_1.0.14828.8_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "intel-compute-runtime_igc-opencl_x86_64",
        sha256 = "133e1981121d60d3b83c0cddee72aaa12195cdf8cd576516fefe4db19420fffe",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/133e1981121d60d3b83c0cddee72aaa12195cdf8cd576516fefe4db19420fffe.deb", "https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-opencl_1.0.14828.8_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "intel-compute-runtime_level-zero-gpu_x86_64",
        sha256 = "9500b20e703497eceaf923aace5311f0c75ae77ae2012c7e2c2c117896e56955",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9500b20e703497eceaf923aace5311f0c75ae77ae2012c7e2c2c117896e56955.deb", "https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu_1.3.26918.9_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "intel-compute-runtime_libigdgmm_x86_64",
        sha256 = "de6ed9333a74506de43993eb14c8e0e0eb23314c42b03af66fe9c3df24a8973d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/de6ed9333a74506de43993eb14c8e0e0eb23314c42b03af66fe9c3df24a8973d.deb", "https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/libigdgmm12_22.3.0_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "intel-compute-runtime_opencl-icd_x86_64",
        sha256 = "5de673d74e8d2e2a62fa739e2476f3a1a44be293e7593536da960dc0ed4678af",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5de673d74e8d2e2a62fa739e2476f3a1a44be293e7593536da960dc0ed4678af.deb", "https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd_23.30.26918.9_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-cccl-11-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "1306a4ca4f83024bd3d02f24a2c911f69dd02c2ff6b6ef386a527e5bd717ff01",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1306a4ca4f83024bd3d02f24a2c911f69dd02c2ff6b6ef386a527e5bd717ff01.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-thrust/cuda-cccl-11-4_11.4.298-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-cudart-11-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "488274208dc7bc2c0b0c803277926510b5e586ecc2b239a695978158da191ccd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/488274208dc7bc2c0b0c803277926510b5e586ecc2b239a695978158da191ccd.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-cudart/cuda-cudart-11-4_11.4.298-1_arm64.deb"],
        deps = ["@jetson_cuda-toolkit-11-4-config-common_aarch64//:all_files", "@jetson_cuda-toolkit-11-config-common_aarch64//:all_files", "@jetson_cuda-toolkit-config-common_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-cudart-dev-11-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "fbeb7d4ea372feaeaed88780490981cd239ae3a3fcf63bc83d87a7288902b9ba",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fbeb7d4ea372feaeaed88780490981cd239ae3a3fcf63bc83d87a7288902b9ba.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-cudart/cuda-cudart-dev-11-4_11.4.298-1_arm64.deb"],
        deps = ["@jetson_cuda-cccl-11-4_aarch64//:all_files", "@jetson_cuda-driver-dev-11-4_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-driver-dev-11-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8366c37ee1088cc27cb48b572de612bb2f1df5a4598c7b59a6a0aeebc0c8bede",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8366c37ee1088cc27cb48b572de612bb2f1df5a4598c7b59a6a0aeebc0c8bede.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-cudart/cuda-driver-dev-11-4_11.4.298-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-nvcc-11-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "17899cc7c04d3766f768c5a41094d09c393043165efe86732ee09ea4f83479cc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/17899cc7c04d3766f768c5a41094d09c393043165efe86732ee09ea4f83479cc.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-nvcc/cuda-nvcc-11-4_11.4.315-1_arm64.deb"],
        deps = ["@jetson_cuda-cudart-dev-11-4_aarch64//:all_files", "@ubuntu2004_build-essential_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-nvrtc-11-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f0e4150e89624d85dc6aaaec09fad97e36d285212f6765b24db9e69b85c2f0b0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f0e4150e89624d85dc6aaaec09fad97e36d285212f6765b24db9e69b85c2f0b0.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-nvrtc/cuda-nvrtc-11-4_11.4.300-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-nvrtc-dev-11-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "204caec1733d39008af5840e22c4e620aff5ee44ecb329fab5b57d6eeea9e06a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/204caec1733d39008af5840e22c4e620aff5ee44ecb329fab5b57d6eeea9e06a.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-nvrtc/cuda-nvrtc-dev-11-4_11.4.300-1_arm64.deb"],
        deps = ["@jetson_cuda-nvrtc-11-4_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-toolkit-11-4-config-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "39d1aee28677c015029972737fdc05bfa95c8e6738470821465bbc7dac5ef270",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/39d1aee28677c015029972737fdc05bfa95c8e6738470821465bbc7dac5ef270.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-cudart/cuda-toolkit-11-4-config-common_11.4.298-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-toolkit-11-config-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "fc68012eed2562378187a120b6df79a5895ba490ef0706714f3071087de89805",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fc68012eed2562378187a120b6df79a5895ba490ef0706714f3071087de89805.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-cudart/cuda-toolkit-11-config-common_11.4.298-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_cuda-toolkit-config-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f04b8db43032a67ffd092f855ed1e7dd3ce4647370f3b85c87ef1d593f460404",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f04b8db43032a67ffd092f855ed1e7dd3ce4647370f3b85c87ef1d593f460404.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cuda-cudart/cuda-toolkit-config-common_11.4.298-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_libcublas-11-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "53de2203b45dc2e47472c9a48e2dd6c5d11b78a9ee94a1d8e5707e11feffe703",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/53de2203b45dc2e47472c9a48e2dd6c5d11b78a9ee94a1d8e5707e11feffe703.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/libc/libcublas/libcublas-11-4_11.6.6.84-1_arm64.deb"],
        deps = ["@jetson_cuda-toolkit-11-4-config-common_aarch64//:all_files", "@jetson_cuda-toolkit-11-config-common_aarch64//:all_files", "@jetson_cuda-toolkit-config-common_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_libcublas-dev-11-4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8db159ef73ccaa926f327d324704dcea157d734e0a70c3df0e12d6367ca2d48a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8db159ef73ccaa926f327d324704dcea157d734e0a70c3df0e12d6367ca2d48a.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/libc/libcublas/libcublas-dev-11-4_11.6.6.84-1_arm64.deb"],
        deps = ["@jetson_libcublas-11-4_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_libcudnn8-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f23632cb15ba1db209811086b5c31207d2da49478402422b9d9aaa9fe9dc0512",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f23632cb15ba1db209811086b5c31207d2da49478402422b9d9aaa9fe9dc0512.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cudnn/libcudnn8-dev_8.6.0.166-1+cuda11.4_arm64.deb"],
        deps = ["@jetson_libcudnn8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_libcudnn8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4a679b9676d4d1bfd2d7a3572eefc916706219ad006e38a8fd2377067fd635ee",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4a679b9676d4d1bfd2d7a3572eefc916706219ad006e38a8fd2377067fd635ee.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/c/cudnn/libcudnn8_8.6.0.166-1+cuda11.4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_libnvinfer-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "14ebffd144def765fdb3560c342984a9266761f3dbe7b3f8034f0fae7908360b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/14ebffd144def765fdb3560c342984a9266761f3dbe7b3f8034f0fae7908360b.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/t/tensorrt/libnvinfer-dev_8.5.2-1+cuda11.4_arm64.deb"],
        deps = ["@jetson_cuda-nvrtc-dev-11-4_aarch64//:all_files", "@jetson_libcublas-dev-11-4_aarch64//:all_files", "@jetson_libcudnn8-dev_aarch64//:all_files", "@jetson_libnvinfer8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_libnvinfer-plugin8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3026cff562d367e35ad25f7d1dde555eea63283317784baa03dd23ef96616db1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3026cff562d367e35ad25f7d1dde555eea63283317784baa03dd23ef96616db1.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/t/tensorrt/libnvinfer-plugin8_8.5.2-1+cuda11.4_arm64.deb"],
        deps = ["@jetson_libnvinfer8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_libnvinfer8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "55012013528106fca0fb0fd3dbf73a54b8e441567fe7144d1523e5609f036c31",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/55012013528106fca0fb0fd3dbf73a54b8e441567fe7144d1523e5609f036c31.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/t/tensorrt/libnvinfer8_8.5.2-1+cuda11.4_arm64.deb"],
        deps = ["@jetson_cuda-nvrtc-11-4_aarch64//:all_files", "@jetson_libcublas-11-4_aarch64//:all_files", "@jetson_libcudnn8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_libnvonnxparsers-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "73c303e0196be808aa61c0902d2e4168c6a5e728f06749dddc3a313df0bb732d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/73c303e0196be808aa61c0902d2e4168c6a5e728f06749dddc3a313df0bb732d.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/t/tensorrt/libnvonnxparsers-dev_8.5.2-1+cuda11.4_arm64.deb"],
        deps = ["@jetson_libnvonnxparsers8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_libnvonnxparsers8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7cd2c461c0ef020853c4df618d0e57ca33bb889d019a691696dba293ac11bbc8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7cd2c461c0ef020853c4df618d0e57ca33bb889d019a691696dba293ac11bbc8.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/t/tensorrt/libnvonnxparsers8_8.5.2-1+cuda11.4_arm64.deb"],
        deps = ["@jetson_libnvinfer8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-camera_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d03d2f5baa111681aa3115e40e24572fba562e88b499a27f525ccf751fb36701",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d03d2f5baa111681aa3115e40e24572fba562e88b499a27f525ccf751fb36701.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-camera/nvidia-l4t-camera_35.4.1-20230801124926_arm64.deb"],
        deps = ["@jetson_nvidia-l4t-core_aarch64//:all_files", "@jetson_nvidia-l4t-cuda_aarch64//:all_files", "@jetson_nvidia-l4t-multimedia-utils_aarch64//:all_files", "@jetson_nvidia-l4t-multimedia_aarch64//:all_files", "@ubuntu2004_libcairo2_aarch64//:all_files", "@ubuntu2004_libegl1-mesa_aarch64//:all_files", "@ubuntu2004_libegl1_aarch64//:all_files", "@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libgcc1_aarch64//:all_files", "@ubuntu2004_libgles2_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libglvnd0_aarch64//:all_files", "@ubuntu2004_libgtk-3-0_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-core_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e0f5e1af4b2c0b00530c7d49187d35a596d6b938ad9055459cd10c072d3b20e2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e0f5e1af4b2c0b00530c7d49187d35a596d6b938ad9055459cd10c072d3b20e2.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-core/nvidia-l4t-core_35.4.1-20230801124926_arm64.deb"],
        deps = ["@ubuntu2004_libegl1_aarch64//:all_files", "@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libgcc1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-cuda_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f1131e207bebff0ef0ec4b14ef09ac7ee1e541bf7f79a24aae1c19c99885c2ba",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f1131e207bebff0ef0ec4b14ef09ac7ee1e541bf7f79a24aae1c19c99885c2ba.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-cuda/nvidia-l4t-cuda_35.4.1-20230801124926_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-init_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3b1cd6cc764fe71e8a36509bc7c4556ea16c50ddbd0963740a359af813d3ff7f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3b1cd6cc764fe71e8a36509bc7c4556ea16c50ddbd0963740a359af813d3ff7f.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-init/nvidia-l4t-init_35.4.1-20230801124926_arm64.deb"],
        deps = ["@ubuntu2004_bridge-utils_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-jetson-multimedia-api_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a28d46509bbe2c2f0dd40c9e43854b5cf95c33ac04502176c68f3c2f1cd7883e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a28d46509bbe2c2f0dd40c9e43854b5cf95c33ac04502176c68f3c2f1cd7883e.deb", "https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-jetson-multimedia-api/nvidia-l4t-jetson-multimedia-api_35.4.1-20230801124926_arm64.deb"],
        deps = ["@jetson_cuda-cudart-11-4_aarch64//:all_files", "@jetson_cuda-cudart-dev-11-4_aarch64//:all_files", "@jetson_nvidia-l4t-multimedia-utils_aarch64//:all_files", "@jetson_nvidia-l4t-multimedia_aarch64//:all_files", "@ubuntu2004_libglvnd-dev_aarch64//:all_files", "@ubuntu2004_libx11-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-kernel-headers_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "856cb87a029b51116d4100f833b524f9849894e24ad013d624ced60a2cfa1ce1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/856cb87a029b51116d4100f833b524f9849894e24ad013d624ced60a2cfa1ce1.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-kernel-headers/nvidia-l4t-kernel-headers_5.10.120-tegra-35.4.1-20230801124926_arm64.deb"],
        deps = ["@jetson_nvidia-l4t-kernel_aarch64//:all_files", "@ubuntu2004_libssl1.1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-kernel_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "47b59b46687d9f69ab6fba8780241c41780a28a581819bbbf12487294f6a6d2c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/47b59b46687d9f69ab6fba8780241c41780a28a581819bbbf12487294f6a6d2c.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-kernel/nvidia-l4t-kernel_5.10.120-tegra-35.4.1-20230801124926_arm64.deb"],
        deps = ["@jetson_nvidia-l4t-init_aarch64//:all_files", "@jetson_nvidia-l4t-tools_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-multimedia-utils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f390756fa416f13285ec9647499baa3d9eaad262b9932257aafdff46f56c9580",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f390756fa416f13285ec9647499baa3d9eaad262b9932257aafdff46f56c9580.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-multimedia-utils/nvidia-l4t-multimedia-utils_35.4.1-20230801124926_arm64.deb"],
        deps = ["@ubuntu2004_libcairo2_aarch64//:all_files", "@ubuntu2004_libdatrie1_aarch64//:all_files", "@ubuntu2004_libegl1_aarch64//:all_files", "@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libfontconfig1_aarch64//:all_files", "@ubuntu2004_libgcc1_aarch64//:all_files", "@ubuntu2004_libgles2_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libglvnd0_aarch64//:all_files", "@ubuntu2004_libharfbuzz0b_aarch64//:all_files", "@ubuntu2004_libpangoft2-1.0-0_aarch64//:all_files", "@ubuntu2004_libpcre3_aarch64//:all_files", "@ubuntu2004_libpixman-1-0_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxau6_aarch64//:all_files", "@ubuntu2004_libxcb1_aarch64//:all_files", "@ubuntu2004_libxdmcp6_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files", "@ubuntu2004_libxrender1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-multimedia_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "14113a0047529bfbb53eb229cf5c25b60a85a3fd32defa371eb68bd824647fe8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/14113a0047529bfbb53eb229cf5c25b60a85a3fd32defa371eb68bd824647fe8.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-multimedia/nvidia-l4t-multimedia_35.4.1-20230801124926_arm64.deb"],
        deps = ["@jetson_nvidia-l4t-multimedia-utils_aarch64//:all_files", "@jetson_nvidia-l4t-nvsci_aarch64//:all_files", "@ubuntu2004_libasound2_aarch64//:all_files", "@ubuntu2004_libcairo2_aarch64//:all_files", "@ubuntu2004_libdatrie1_aarch64//:all_files", "@ubuntu2004_libegl1_aarch64//:all_files", "@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libfontconfig1_aarch64//:all_files", "@ubuntu2004_libgcc1_aarch64//:all_files", "@ubuntu2004_libgles2_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libglvnd0_aarch64//:all_files", "@ubuntu2004_libgstreamer-plugins-base1.0-0_aarch64//:all_files", "@ubuntu2004_libgstreamer1.0-0_aarch64//:all_files", "@ubuntu2004_libharfbuzz0b_aarch64//:all_files", "@ubuntu2004_libpango-1.0-0_aarch64//:all_files", "@ubuntu2004_libpangocairo-1.0-0_aarch64//:all_files", "@ubuntu2004_libpangoft2-1.0-0_aarch64//:all_files", "@ubuntu2004_libpcre3_aarch64//:all_files", "@ubuntu2004_libpixman-1-0_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxau6_aarch64//:all_files", "@ubuntu2004_libxcb1_aarch64//:all_files", "@ubuntu2004_libxdmcp6_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files", "@ubuntu2004_libxrender1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-nvsci_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9411218cad37f7bdc84eccd9d556bda8fe340a93df8f349f12c5c3ca53839f8d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9411218cad37f7bdc84eccd9d556bda8fe340a93df8f349f12c5c3ca53839f8d.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-nvsci/nvidia-l4t-nvsci_35.4.1-20230801124926_arm64.deb"],
        deps = ["@ubuntu2004_libgcc1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "jetson_nvidia-l4t-tools_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "83d719a866b4477a4c98efc27e46c8694f6fc99402954e027dc208990d2205fb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/83d719a866b4477a4c98efc27e46c8694f6fc99402954e027dc208990d2205fb.deb", "https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-tools/nvidia-l4t-tools_35.4.1-20230801124926_arm64.deb"],
        deps = ["@ubuntu2004_libnl-genl-3-200_aarch64//:all_files", "@ubuntu2004_libnl-route-3-200_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-cccl-12-3_x86_64",
        sha256 = "d00c678373128e45661fbfe1791e598fc7c5c8ce8016834f7beac90c148bf923",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d00c678373128e45661fbfe1791e598fc7c5c8ce8016834f7beac90c148bf923.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-cccl-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-command-line-tools-12-3_x86_64",
        sha256 = "371015766f8411deb570cfaa52dd0837eaca2f50b02fe0642b1ca43a4f81a579",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/371015766f8411deb570cfaa52dd0837eaca2f50b02fe0642b1ca43a4f81a579.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-command-line-tools-12-3_12.3.2-1_amd64.deb"],
        deps = ["@nvidia_cuda-cupti-dev-12-3_x86_64//:all_files", "@nvidia_cuda-gdb-12-3_x86_64//:all_files", "@nvidia_cuda-nvdisasm-12-3_x86_64//:all_files", "@nvidia_cuda-nvprof-12-3_x86_64//:all_files", "@nvidia_cuda-nvtx-12-3_x86_64//:all_files", "@nvidia_cuda-sanitizer-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-compiler-12-3_x86_64",
        sha256 = "b027e7549fbe1b5258ac81ea85a86ee9e74f9636edb9e96520249cf9cce252a8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b027e7549fbe1b5258ac81ea85a86ee9e74f9636edb9e96520249cf9cce252a8.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-compiler-12-3_12.3.2-1_amd64.deb"],
        deps = ["@nvidia_cuda-cuobjdump-12-3_x86_64//:all_files", "@nvidia_cuda-cuxxfilt-12-3_x86_64//:all_files", "@nvidia_cuda-nvcc-12-3_x86_64//:all_files", "@nvidia_cuda-nvprune-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-crt-12-3_x86_64",
        sha256 = "33c8a03beae7f2790d99cbf73158ab5e67db71a560cfee7611893fa27f4acdac",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/33c8a03beae7f2790d99cbf73158ab5e67db71a560cfee7611893fa27f4acdac.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-crt-12-3_12.3.107-1_amd64.deb"],
        deps = ["@nvidia_cuda-cudart-dev-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-cudart-12-3_x86_64",
        sha256 = "fd3fce7cdec43675c452bea3060f50cad69562e7aa85f75b146532c45d3b9142",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fd3fce7cdec43675c452bea3060f50cad69562e7aa85f75b146532c45d3b9142.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-cudart-12-3_12.3.101-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-cudart-dev-12-3_x86_64",
        sha256 = "0c88aa6232503568f0d5468285cf2395228c24266fbdab11f104980f8ef4926a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0c88aa6232503568f0d5468285cf2395228c24266fbdab11f104980f8ef4926a.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-cudart-dev-12-3_12.3.101-1_amd64.deb"],
        deps = ["@nvidia_cuda-cccl-12-3_x86_64//:all_files", "@nvidia_cuda-cudart-12-3_x86_64//:all_files", "@nvidia_cuda-driver-dev-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-cuobjdump-12-3_x86_64",
        sha256 = "345c85e3824f167cac5ef56b1d55cad82516ba7ae09dead0f1af5f63fa1270fd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/345c85e3824f167cac5ef56b1d55cad82516ba7ae09dead0f1af5f63fa1270fd.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-cuobjdump-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-cupti-12-3_x86_64",
        sha256 = "ced15e5958757942b7a6014ad8b5d91e9f71be2e29ab81bc10d99a3910e90bd1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ced15e5958757942b7a6014ad8b5d91e9f71be2e29ab81bc10d99a3910e90bd1.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-cupti-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-cupti-dev-12-3_x86_64",
        sha256 = "113cc5411cf217d8c782867cb3da5be1bd52b8facf1e59396b9ef355e4abf773",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/113cc5411cf217d8c782867cb3da5be1bd52b8facf1e59396b9ef355e4abf773.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-cupti-dev-12-3_12.3.101-1_amd64.deb"],
        deps = ["@nvidia_cuda-cupti-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-cuxxfilt-12-3_x86_64",
        sha256 = "713c0d470f8a0b3dfb6ed46a8a3975dae27ccaaecd8166ebac269d27907dbccd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/713c0d470f8a0b3dfb6ed46a8a3975dae27ccaaecd8166ebac269d27907dbccd.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-cuxxfilt-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-documentation-12-3_x86_64",
        sha256 = "df1013ac3b3bbc0be3cdfc91edc6009f28a84f8fb2e9b36e8008b8c2e1d51718",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/df1013ac3b3bbc0be3cdfc91edc6009f28a84f8fb2e9b36e8008b8c2e1d51718.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-documentation-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-driver-dev-12-3_x86_64",
        sha256 = "865b16610fca1f4f6234bcd69d6ea5ecffa3695d21bbc61b3695b9a03c3b3c4f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/865b16610fca1f4f6234bcd69d6ea5ecffa3695d21bbc61b3695b9a03c3b3c4f.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-driver-dev-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-gdb-12-3_x86_64",
        sha256 = "6a7b568fc632d557cda62c6b5aca7b2b9067c30607014fc9c072b8583407f6ae",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6a7b568fc632d557cda62c6b5aca7b2b9067c30607014fc9c072b8583407f6ae.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-gdb-12-3_12.3.101-1_amd64.deb"],
        deps = ["@nvidia_cuda-cuobjdump-12-3_x86_64//:all_files", "@nvidia_cuda-nvdisasm-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-libraries-12-3_x86_64",
        sha256 = "6dbf9d34ecfce50d5eb871d4d98421786153590e1935a32103fe2c843c1849a2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6dbf9d34ecfce50d5eb871d4d98421786153590e1935a32103fe2c843c1849a2.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-libraries-12-3_12.3.2-1_amd64.deb"],
        deps = ["@nvidia_cuda-cudart-12-3_x86_64//:all_files", "@nvidia_cuda-nvrtc-12-3_x86_64//:all_files", "@nvidia_cuda-opencl-12-3_x86_64//:all_files", "@nvidia_libcublas-12-3_x86_64//:all_files", "@nvidia_libcufft-12-3_x86_64//:all_files", "@nvidia_libcurand-12-3_x86_64//:all_files", "@nvidia_libcusolver-12-3_x86_64//:all_files", "@nvidia_libcusparse-12-3_x86_64//:all_files", "@nvidia_libnpp-12-3_x86_64//:all_files", "@nvidia_libnvjitlink-12-3_x86_64//:all_files", "@nvidia_libnvjpeg-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-libraries-dev-12-3_x86_64",
        sha256 = "5eedb9eea413bcd0a9f7c327667c03b89fe9cff5a47d171ccb5d44b904875b5b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5eedb9eea413bcd0a9f7c327667c03b89fe9cff5a47d171ccb5d44b904875b5b.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-libraries-dev-12-3_12.3.2-1_amd64.deb"],
        deps = ["@nvidia_cuda-cccl-12-3_x86_64//:all_files", "@nvidia_cuda-cudart-dev-12-3_x86_64//:all_files", "@nvidia_cuda-driver-dev-12-3_x86_64//:all_files", "@nvidia_cuda-nvrtc-dev-12-3_x86_64//:all_files", "@nvidia_cuda-opencl-dev-12-3_x86_64//:all_files", "@nvidia_cuda-profiler-api-12-3_x86_64//:all_files", "@nvidia_libcublas-dev-12-3_x86_64//:all_files", "@nvidia_libcufft-dev-12-3_x86_64//:all_files", "@nvidia_libcufile-dev-12-3_x86_64//:all_files", "@nvidia_libcurand-dev-12-3_x86_64//:all_files", "@nvidia_libcusolver-dev-12-3_x86_64//:all_files", "@nvidia_libcusparse-dev-12-3_x86_64//:all_files", "@nvidia_libnpp-dev-12-3_x86_64//:all_files", "@nvidia_libnvjitlink-dev-12-3_x86_64//:all_files", "@nvidia_libnvjpeg-dev-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-nvcc-12-3_x86_64",
        sha256 = "2c9c2a62651dbd500435c881416f469357ec43f6f8eebfdd335588e0d02e5f33",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2c9c2a62651dbd500435c881416f469357ec43f6f8eebfdd335588e0d02e5f33.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-nvcc-12-3_12.3.107-1_amd64.deb"],
        deps = ["@nvidia_cuda-crt-12-3_x86_64//:all_files", "@nvidia_cuda-cudart-dev-12-3_x86_64//:all_files", "@nvidia_cuda-nvvm-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-nvdisasm-12-3_x86_64",
        sha256 = "1e3b7715bd72c59216bfe8753667898841047c609347bd38a6829e4ff309f972",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1e3b7715bd72c59216bfe8753667898841047c609347bd38a6829e4ff309f972.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-nvdisasm-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-nvml-dev-12-3_x86_64",
        sha256 = "1ce1d191736e3c4cc846162b932776e38572f993409dcbef429c8c42555008a1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1ce1d191736e3c4cc846162b932776e38572f993409dcbef429c8c42555008a1.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-nvml-dev-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-nvprof-12-3_x86_64",
        sha256 = "7028304f0d9dd46eafb2905b65349f5df07eb35cc1edfca59283544299dfb12b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7028304f0d9dd46eafb2905b65349f5df07eb35cc1edfca59283544299dfb12b.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-nvprof-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-nvprune-12-3_x86_64",
        sha256 = "93ccebdafa087f7b74bc3f1d8fa314558d479bfa66136ad91a7a11c56d943dfe",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/93ccebdafa087f7b74bc3f1d8fa314558d479bfa66136ad91a7a11c56d943dfe.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-nvprune-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-nvrtc-12-3_x86_64",
        sha256 = "f36740f528a5cd3089725478dc46fc30293e249043ca7d3cd009b3e39ae5de55",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f36740f528a5cd3089725478dc46fc30293e249043ca7d3cd009b3e39ae5de55.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-nvrtc-12-3_12.3.107-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-nvrtc-dev-12-3_x86_64",
        sha256 = "123a3aaa41995192b529ddb4c6ff35d56a024c6dbbcfda571d8f065d11621de5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/123a3aaa41995192b529ddb4c6ff35d56a024c6dbbcfda571d8f065d11621de5.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-nvrtc-dev-12-3_12.3.107-1_amd64.deb"],
        deps = ["@nvidia_cuda-nvrtc-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-nvtx-12-3_x86_64",
        sha256 = "09a4c8d0e6ff05a5ba94812d74bb534209f7a8adb0b290767109dd351b705b12",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/09a4c8d0e6ff05a5ba94812d74bb534209f7a8adb0b290767109dd351b705b12.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-nvtx-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-nvvm-12-3_x86_64",
        sha256 = "66da8086b45a2b1c9266ee51b53b0b1fb403ec5cba764b8a37396d6b2b3e1c9f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/66da8086b45a2b1c9266ee51b53b0b1fb403ec5cba764b8a37396d6b2b3e1c9f.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-nvvm-12-3_12.3.107-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-opencl-12-3_x86_64",
        sha256 = "b6d5dcf61831ef4e2c176d451eed36803bb9d19d7801b929edd4cddfdc3b8b36",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b6d5dcf61831ef4e2c176d451eed36803bb9d19d7801b929edd4cddfdc3b8b36.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-opencl-12-3_12.3.101-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-opencl-dev-12-3_x86_64",
        sha256 = "c07c2bee317bb110482e71aa9bd727640fd4ed746ef4537956f5dde6b725d70e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c07c2bee317bb110482e71aa9bd727640fd4ed746ef4537956f5dde6b725d70e.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-opencl-dev-12-3_12.3.101-1_amd64.deb"],
        deps = ["@nvidia_cuda-opencl-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-profiler-api-12-3_x86_64",
        sha256 = "96862c14435fe29c142861543cf3edc1b393c8198f5449f8902db4066e6c4791",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/96862c14435fe29c142861543cf3edc1b393c8198f5449f8902db4066e6c4791.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-profiler-api-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-sanitizer-12-3_x86_64",
        sha256 = "ed51f75f2a70929b71e714fa42dc01cc9db39835eb9ec7b9f03b7fcd624dea66",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ed51f75f2a70929b71e714fa42dc01cc9db39835eb9ec7b9f03b7fcd624dea66.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-sanitizer-12-3_12.3.101-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-toolkit-12-3-config-common_x86_64",
        sha256 = "b14276527b4f19d94cd43e789ef6e76ca693ff438140bf471958ba864bf3891e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b14276527b4f19d94cd43e789ef6e76ca693ff438140bf471958ba864bf3891e.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-toolkit-12-3-config-common_12.3.101-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-toolkit-12-3_x86_64",
        sha256 = "e20c7f87d74caf949f3d0876fd32bae13e1321ab52f83630a1e43cf6ae179ee1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e20c7f87d74caf949f3d0876fd32bae13e1321ab52f83630a1e43cf6ae179ee1.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-toolkit-12-3_12.3.2-1_amd64.deb"],
        deps = ["@nvidia_cuda-compiler-12-3_x86_64//:all_files", "@nvidia_cuda-documentation-12-3_x86_64//:all_files", "@nvidia_cuda-libraries-12-3_x86_64//:all_files", "@nvidia_cuda-libraries-dev-12-3_x86_64//:all_files", "@nvidia_cuda-nvml-dev-12-3_x86_64//:all_files", "@nvidia_cuda-tools-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-toolkit-12-config-common_x86_64",
        sha256 = "1f358b73dd428f1c6d086e2b79a09a44f9a213ea22a25b8ea699695b64c92d65",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1f358b73dd428f1c6d086e2b79a09a44f9a213ea22a25b8ea699695b64c92d65.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-toolkit-12-config-common_12.6.37-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-toolkit-config-common_x86_64",
        sha256 = "f46a9437302a6ed20166dbb11fdafb617a19fccb6162cf83f5956a699e1f953d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f46a9437302a6ed20166dbb11fdafb617a19fccb6162cf83f5956a699e1f953d.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-toolkit-config-common_12.6.37-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_cuda-tools-12-3_x86_64",
        sha256 = "f4ccf6f164fd3ee474e31cf1e3e675326471b62196a17b66eca26de8f0a255a9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f4ccf6f164fd3ee474e31cf1e3e675326471b62196a17b66eca26de8f0a255a9.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./cuda-tools-12-3_12.3.2-1_amd64.deb"],
        deps = ["@nvidia_cuda-command-line-tools-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcublas-12-3_x86_64",
        sha256 = "92c89dd47e6961aeec15bf93bb7f8663a6188e6f91b4ef1ac2478c9b24e0fc8b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/92c89dd47e6961aeec15bf93bb7f8663a6188e6f91b4ef1ac2478c9b24e0fc8b.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcublas-12-3_12.3.4.1-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcublas-dev-12-3_x86_64",
        sha256 = "56e55f9e0d29537668106e9f296bbcde8f383963cf9f7c122b41c1019b8a4220",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/56e55f9e0d29537668106e9f296bbcde8f383963cf9f7c122b41c1019b8a4220.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcublas-dev-12-3_12.3.4.1-1_amd64.deb"],
        deps = ["@nvidia_libcublas-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcufft-12-3_x86_64",
        sha256 = "ac89b9b7ae4c4df8eb79d65c1abfd321a47cdf3590f81c2f2f4aa92c50465565",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ac89b9b7ae4c4df8eb79d65c1abfd321a47cdf3590f81c2f2f4aa92c50465565.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcufft-12-3_11.0.12.1-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcufft-dev-12-3_x86_64",
        sha256 = "e9746f270d0f296f0b5f8db3d0f821c898283c57e03f04ff08122b5482100f91",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e9746f270d0f296f0b5f8db3d0f821c898283c57e03f04ff08122b5482100f91.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcufft-dev-12-3_11.0.12.1-1_amd64.deb"],
        deps = ["@nvidia_libcufft-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcufile-dev-12-3_x86_64",
        sha256 = "bfa0192ef2e0eaa510a9c0095a5a016cf14185d2a8a42f83f38c30d7b60a6365",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bfa0192ef2e0eaa510a9c0095a5a016cf14185d2a8a42f83f38c30d7b60a6365.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcufile-dev-12-3_1.8.1.2-1_amd64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcurand-12-3_x86_64",
        sha256 = "7cf301a647200abee73815f7573cd3e18d282237432ed2633517d9de88f1de1e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7cf301a647200abee73815f7573cd3e18d282237432ed2633517d9de88f1de1e.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcurand-12-3_10.3.4.107-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcurand-dev-12-3_x86_64",
        sha256 = "9e4a63d06aa5d94bb1bbb8bc4fdcd703edde168b8ce4391731643a4d1b2e3589",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9e4a63d06aa5d94bb1bbb8bc4fdcd703edde168b8ce4391731643a4d1b2e3589.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcurand-dev-12-3_10.3.4.107-1_amd64.deb"],
        deps = ["@nvidia_libcurand-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcusolver-12-3_x86_64",
        sha256 = "319a12863060ed881ad6923a64856fee27bcd0727ae7abf06e79ffa85da2176d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/319a12863060ed881ad6923a64856fee27bcd0727ae7abf06e79ffa85da2176d.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcusolver-12-3_11.5.4.101-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcusolver-dev-12-3_x86_64",
        sha256 = "f971c8d9d9a635b8df1b8dd2d82bc4cefcde8c4733cd32b55f2f93b5ebe9362a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f971c8d9d9a635b8df1b8dd2d82bc4cefcde8c4733cd32b55f2f93b5ebe9362a.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcusolver-dev-12-3_11.5.4.101-1_amd64.deb"],
        deps = ["@nvidia_libcusolver-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcusparse-12-3_x86_64",
        sha256 = "0f747b1b42aef83d2857031f28667123a378e0f7160e3a7d818689ae0b1e3e1b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0f747b1b42aef83d2857031f28667123a378e0f7160e3a7d818689ae0b1e3e1b.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcusparse-12-3_12.2.0.103-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libcusparse-dev-12-3_x86_64",
        sha256 = "e5ed533dfc04961deeb2e525b22d349498653f95ed530b2d06e9dd436934a86c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e5ed533dfc04961deeb2e525b22d349498653f95ed530b2d06e9dd436934a86c.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libcusparse-dev-12-3_12.2.0.103-1_amd64.deb"],
        deps = ["@nvidia_libcusparse-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libnpp-12-3_x86_64",
        extra_symlinks = {
            "usr/local/cuda": "/usr/local/cuda-12.3",
        },
        sha256 = "ae6376d0a7817c150357cab152e167265b2ec793627943daf57eee8cd35faf67",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ae6376d0a7817c150357cab152e167265b2ec793627943daf57eee8cd35faf67.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libnpp-12-3_12.2.3.2-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libnpp-dev-12-3_x86_64",
        sha256 = "3ce918f118cf9c3cea4ce56cb815e687245f56ab839b27e13cf5c0289e7a2f7f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3ce918f118cf9c3cea4ce56cb815e687245f56ab839b27e13cf5c0289e7a2f7f.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libnpp-dev-12-3_12.2.3.2-1_amd64.deb"],
        deps = ["@nvidia_libnpp-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libnvjitlink-12-3_x86_64",
        sha256 = "3584a84bff54e3896a67e92b75f96ed75ff5823ed4ce704a172a55e8959a4c71",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3584a84bff54e3896a67e92b75f96ed75ff5823ed4ce704a172a55e8959a4c71.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libnvjitlink-12-3_12.3.101-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libnvjitlink-dev-12-3_x86_64",
        sha256 = "78f864fccc0dbb470861b80e3b7ca28a636654f8dddc3836019ed8c9ced689bf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/78f864fccc0dbb470861b80e3b7ca28a636654f8dddc3836019ed8c9ced689bf.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libnvjitlink-dev-12-3_12.3.101-1_amd64.deb"],
        deps = ["@nvidia_libnvjitlink-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libnvjpeg-12-3_x86_64",
        sha256 = "7556266a34c724281f1480111f986e2e1798e905302d5f86bdc27f985d88da19",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7556266a34c724281f1480111f986e2e1798e905302d5f86bdc27f985d88da19.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libnvjpeg-12-3_12.3.0.81-1_amd64.deb"],
        deps = ["@nvidia_cuda-toolkit-12-3-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-12-config-common_x86_64//:all_files", "@nvidia_cuda-toolkit-config-common_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "nvidia_libnvjpeg-dev-12-3_x86_64",
        sha256 = "bb9ada9920af13c8e83c064342c6bc326321173a3022097b74cda3191e30a7f9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bb9ada9920af13c8e83c064342c6bc326321173a3022097b74cda3191e30a7f9.deb", "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/./libnvjpeg-dev-12-3_12.3.0.81-1_amd64.deb"],
        deps = ["@nvidia_libnvjpeg-12-3_x86_64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_adduser_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "5f7ea9d1d52a2a9c349468f89d160230e21c8542faed1b1a97c23bce873e17b4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5f7ea9d1d52a2a9c349468f89d160230e21c8542faed1b1a97c23bce873e17b4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/adduser/adduser_3.118ubuntu2_all.deb"],
        deps = ["@ubuntu2004_debconf_aarch64//:all_files", "@ubuntu2004_passwd_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_adwaita-icon-theme_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b30867618d63b66b21761377915f12b64341a464e1834e0dacd87c8d0491abc3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b30867618d63b66b21761377915f12b64341a464e1834e0dacd87c8d0491abc3.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/adwaita-icon-theme/adwaita-icon-theme_3.36.0-1ubuntu1_all.deb"],
        deps = ["@ubuntu2004_gtk-update-icon-cache_aarch64//:all_files", "@ubuntu2004_hicolor-icon-theme_aarch64//:all_files", "@ubuntu2004_librsvg2-common_aarch64//:all_files", "@ubuntu2004_ubuntu-mono_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_base-files_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "252b15b8240184a067146e8688598c55fd2f8d0b8f7243100bde063da25f4cf7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/252b15b8240184a067146e8688598c55fd2f8d0b8f7243100bde063da25f4cf7.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/base-files/base-files_11ubuntu5_arm64.deb"],
        deps = ["@ubuntu2004_gawk_aarch64//:all_files", "@ubuntu2004_libcrypt1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_bash_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2817ffdb13011bbcba0adf0cd7661f575c5c3dd4c9fc136694bf00e46338e27c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2817ffdb13011bbcba0adf0cd7661f575c5c3dd4c9fc136694bf00e46338e27c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/bash/bash_5.0-6ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_base-files_aarch64//:all_files", "@ubuntu2004_debianutils_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_binutils-aarch64-linux-gnu_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "01170efae77aa1d3582fa5569a0e0813f08177e613e7041a1a515ac403fbe9aa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/01170efae77aa1d3582fa5569a0e0813f08177e613e7041a1a515ac403fbe9aa.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/binutils/binutils-aarch64-linux-gnu_2.34-6ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libctf-nobfd0_aarch64//:all_files", "@ubuntu2004_libctf0_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_binutils-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9cbdd2f499e97c09f13a8df349ea96ee0be2a8b2e125071b013618db322e6fd7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9cbdd2f499e97c09f13a8df349ea96ee0be2a8b2e125071b013618db322e6fd7.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/binutils/binutils-common_2.34-6ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_binutils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c55c708736a8695d159438656f8abd64dd56508c2bd64052d97424b67c8a6845",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c55c708736a8695d159438656f8abd64dd56508c2bd64052d97424b67c8a6845.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/binutils/binutils_2.34-6ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_binutils-aarch64-linux-gnu_aarch64//:all_files", "@ubuntu2004_binutils-common_aarch64//:all_files", "@ubuntu2004_libbinutils_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_bridge-utils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a72d4f719c03084af3d435367c01787f665a5b9f68cda9f5c9ec0bf697af76cf",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a72d4f719c03084af3d435367c01787f665a5b9f68cda9f5c9ec0bf697af76cf.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/bridge-utils/bridge-utils_1.6-2ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_build-essential_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9706d895ce47d0bdd7ec1463657e59b63c4ae4f0fcc03365f4226cd67ddfa7fd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9706d895ce47d0bdd7ec1463657e59b63c4ae4f0fcc03365f4226cd67ddfa7fd.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/build-essential/build-essential_12.8ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_dpkg-dev_aarch64//:all_files", "@ubuntu2004_g___aarch64//:all_files", "@ubuntu2004_gcc_aarch64//:all_files", "@ubuntu2004_make_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_bzip2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "5958c36c6f3364f41db06b87acfd0df12cd4007a147954c10aa6f2b8d65fa8e1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5958c36c6f3364f41db06b87acfd0df12cd4007a147954c10aa6f2b8d65fa8e1.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/bzip2/bzip2_1.0.8-2_arm64.deb"],
        deps = ["@ubuntu2004_libbz2-1.0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_ca-certificates_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "978b01e027388615cac0e42a16271328c03dccc18c9b27b985fbabbd5ccf9078",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/978b01e027388615cac0e42a16271328c03dccc18c9b27b985fbabbd5ccf9078.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/c/ca-certificates/ca-certificates_20190110ubuntu1_all.deb"],
        deps = ["@ubuntu2004_debconf_aarch64//:all_files", "@ubuntu2004_openssl_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_coreutils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c80c4af2918f4131292b6d865d35bff225142f5e192f161117670e802035a047",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c80c4af2918f4131292b6d865d35bff225142f5e192f161117670e802035a047.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/c/coreutils/coreutils_8.30-3ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_libacl1_aarch64//:all_files", "@ubuntu2004_libattr1_aarch64//:all_files", "@ubuntu2004_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_cpp-9_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "1fcdf8b70fe621c72b2060c967e1c8e72881b6d19adec2a5363ba48451362a8f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1fcdf8b70fe621c72b2060c967e1c8e72881b6d19adec2a5363ba48451362a8f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-9/cpp-9_9.3.0-10ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_gcc-9-base_aarch64//:all_files", "@ubuntu2004_libgmp10_aarch64//:all_files", "@ubuntu2004_libisl22_aarch64//:all_files", "@ubuntu2004_libmpc3_aarch64//:all_files", "@ubuntu2004_libmpfr6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_cpp_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ddd83313afe0f05c57b321a9a1c7318d8fa022902bce2e4da15ecff8268f4d6b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ddd83313afe0f05c57b321a9a1c7318d8fa022902bce2e4da15ecff8268f4d6b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-defaults/cpp_9.3.0-1ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_cpp-9_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_dbus-user-session_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "744fd2b5ddf1270f0b5a4503e6b50176174e6938c6cc1eb78272cb82a13a7e04",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/744fd2b5ddf1270f0b5a4503e6b50176174e6938c6cc1eb78272cb82a13a7e04.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/dbus/dbus-user-session_1.12.16-2ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_dbus_aarch64//:all_files", "@ubuntu2004_libpam-systemd_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_dbus_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e5637b4a43f298b3710d9a02093eeff1788003a2290c2a269ed1a7b8d696e26a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e5637b4a43f298b3710d9a02093eeff1788003a2290c2a269ed1a7b8d696e26a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/dbus/dbus_1.12.16-2ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_adduser_aarch64//:all_files", "@ubuntu2004_libapparmor1_aarch64//:all_files", "@ubuntu2004_libaudit1_aarch64//:all_files", "@ubuntu2004_libcap-ng0_aarch64//:all_files", "@ubuntu2004_libdbus-1-3_aarch64//:all_files", "@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libselinux1_aarch64//:all_files", "@ubuntu2004_libsystemd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_dconf-gsettings-backend_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c3aad987c31aa48ecbbd434b859ec58f77eeb46817f66f76b387424cb4df5c76",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c3aad987c31aa48ecbbd434b859ec58f77eeb46817f66f76b387424cb4df5c76.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/dconf/dconf-gsettings-backend_0.36.0-1_arm64.deb"],
        deps = ["@ubuntu2004_dconf-service_aarch64//:all_files", "@ubuntu2004_libdconf1_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_dconf-service_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f8caa66fa16ac132a03b6a0a0ecdc4b9da4552fc5e9af7ffe16e1c97c89fd744",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f8caa66fa16ac132a03b6a0a0ecdc4b9da4552fc5e9af7ffe16e1c97c89fd744.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/dconf/dconf-service_0.36.0-1_arm64.deb"],
        deps = ["@ubuntu2004_dbus-user-session_aarch64//:all_files", "@ubuntu2004_libdconf1_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_debconf_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7d639ffea1be8e8f5859ca94a0a09b1e4664d10af421c748b4163b08db515990",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7d639ffea1be8e8f5859ca94a0a09b1e4664d10af421c748b4163b08db515990.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/debconf/debconf_1.5.73_all.deb"],
        deps = ["@ubuntu2004_perl-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_debianutils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b082366aa0885dce8ac1e59296e97fd4fef6aff1c9daa0b1a44fbade85340102",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b082366aa0885dce8ac1e59296e97fd4fef6aff1c9daa0b1a44fbade85340102.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/debianutils/debianutils_4.9.1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_dpkg-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c5163213c0096f4e59e8f34018820b06e60c9994c52561aafb0c821dafe5e35f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c5163213c0096f4e59e8f34018820b06e60c9994c52561aafb0c821dafe5e35f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/dpkg/dpkg-dev_1.19.7ubuntu3_all.deb"],
        deps = ["@ubuntu2004_binutils_aarch64//:all_files", "@ubuntu2004_bzip2_aarch64//:all_files", "@ubuntu2004_libdpkg-perl_aarch64//:all_files", "@ubuntu2004_patch_aarch64//:all_files", "@ubuntu2004_perl_aarch64//:all_files", "@ubuntu2004_tar_aarch64//:all_files", "@ubuntu2004_xz-utils_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_fontconfig-config_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "80b090925c52aff3f4681b361c9823b041c42c57cca58b5baf2541bafc25fed9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/80b090925c52aff3f4681b361c9823b041c42c57cca58b5baf2541bafc25fed9.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/f/fontconfig/fontconfig-config_2.13.1-2ubuntu3_all.deb"],
        deps = ["@ubuntu2004_fonts-dejavu-core_aarch64//:all_files", "@ubuntu2004_ucf_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_fontconfig_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "87e5d450b2c5b042dea9ff08bd5e099acaddf7f53d8c3cfba8718ba6d5583974",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/87e5d450b2c5b042dea9ff08bd5e099acaddf7f53d8c3cfba8718ba6d5583974.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/f/fontconfig/fontconfig_2.13.1-2ubuntu3_arm64.deb"],
        deps = ["@ubuntu2004_fontconfig-config_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_fonts-dejavu-core_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f2b3f7f51e23e0493e8e642c82003fe75cf42bc95fda545cc96b725a69adb515",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f2b3f7f51e23e0493e8e642c82003fe75cf42bc95fda545cc96b725a69adb515.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/f/fonts-dejavu/fonts-dejavu-core_2.37-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_g__-9_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9b2a949f25150e29226566f87b88c2bac236d4d52ae648d2916a123920f16165",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9b2a949f25150e29226566f87b88c2bac236d4d52ae648d2916a123920f16165.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-9/g++-9_9.3.0-10ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_gcc-9-base_aarch64//:all_files", "@ubuntu2004_gcc-9_aarch64//:all_files", "@ubuntu2004_libgmp10_aarch64//:all_files", "@ubuntu2004_libisl22_aarch64//:all_files", "@ubuntu2004_libmpc3_aarch64//:all_files", "@ubuntu2004_libmpfr6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_g___aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "982061e7f9b672dcb0af67d4567777cda6867e59dc01138e5fd281b021834090",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/982061e7f9b672dcb0af67d4567777cda6867e59dc01138e5fd281b021834090.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-defaults/g++_9.3.0-1ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_cpp_aarch64//:all_files", "@ubuntu2004_g__-9_aarch64//:all_files", "@ubuntu2004_gcc-9_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_gawk_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "cb85be7e4710392316a0b33f7be12a8b6b4f57fe5f156c1d150b1ceb595a9739",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cb85be7e4710392316a0b33f7be12a8b6b4f57fe5f156c1d150b1ceb595a9739.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gawk/gawk_5.0.1+dfsg-1_arm64.deb"],
        deps = ["@ubuntu2004_libgmp10_aarch64//:all_files", "@ubuntu2004_libmpfr6_aarch64//:all_files", "@ubuntu2004_libreadline8_aarch64//:all_files", "@ubuntu2004_libsigsegv2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_gcc-10-base_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e3584ff93d238bfe77cc7ad3a8a599bfb452c1a785b09fe04cdfbc384d12e5e0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e3584ff93d238bfe77cc7ad3a8a599bfb452c1a785b09fe04cdfbc384d12e5e0.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/gcc-10-base_10-20200411-0ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_gcc-9-base_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "fd3c523f5be9e389225c9cc7a9f79f35b3fa03055340a2dbc266b1cab386adae",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fd3c523f5be9e389225c9cc7a9f79f35b3fa03055340a2dbc266b1cab386adae.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-9/gcc-9-base_9.3.0-10ubuntu2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_gcc-9_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "5105aea52ca7381641fda3ff820af0280e36ac17421f1241f36b4368a8473b7b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5105aea52ca7381641fda3ff820af0280e36ac17421f1241f36b4368a8473b7b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-9/gcc-9_9.3.0-10ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_binutils_aarch64//:all_files", "@ubuntu2004_cpp-9_aarch64//:all_files", "@ubuntu2004_gcc-9-base_aarch64//:all_files", "@ubuntu2004_libcc1-0_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files", "@ubuntu2004_libgmp10_aarch64//:all_files", "@ubuntu2004_libisl22_aarch64//:all_files", "@ubuntu2004_libmpc3_aarch64//:all_files", "@ubuntu2004_libmpfr6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_gcc_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0e3b2004fa01730d173b7c0e041c610cb4a64a3816ffc2a555ad1814657ac959",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0e3b2004fa01730d173b7c0e041c610cb4a64a3816ffc2a555ad1814657ac959.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-defaults/gcc_9.3.0-1ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_cpp_aarch64//:all_files", "@ubuntu2004_gcc-9_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_glib-networking-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0064a0ecf7811f5ad63db5380fc5d11d6ff8c16e33c8c74f28f4ee13995dc1cd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0064a0ecf7811f5ad63db5380fc5d11d6ff8c16e33c8c74f28f4ee13995dc1cd.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/glib-networking/glib-networking-common_2.64.1-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_glib-networking-services_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f2db9731adfb6a0a35d5fe28ca4d7fb84f3cbef39e62c854a909a835eb007f44",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f2db9731adfb6a0a35d5fe28ca4d7fb84f3cbef39e62c854a909a835eb007f44.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/glib-networking/glib-networking-services_2.64.1-1_arm64.deb"],
        deps = ["@ubuntu2004_glib-networking-common_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_glib-networking_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "04ccde69ace8db26c43e72295a88dd7af9162c4864c403fc2a268157b744fb3d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/04ccde69ace8db26c43e72295a88dd7af9162c4864c403fc2a268157b744fb3d.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/glib-networking/glib-networking_2.64.1-1_arm64.deb"],
        deps = ["@ubuntu2004_glib-networking-common_aarch64//:all_files", "@ubuntu2004_glib-networking-services_aarch64//:all_files", "@ubuntu2004_gsettings-desktop-schemas_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libgnutls30_aarch64//:all_files", "@ubuntu2004_libproxy1v5_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_grep_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8513e7def0e04a43a4a98723fdd2bf17b4df0bdf272252c7c6dc9660a33790d4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8513e7def0e04a43a4a98723fdd2bf17b4df0bdf272252c7c6dc9660a33790d4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/grep/grep_3.4-1_arm64.deb"],
        deps = ["@ubuntu2004_libpcre3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_gsettings-desktop-schemas_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "bf9c1662992a492210e11af477cd3dae879c89a979c67f6821c9af1b3ec8abde",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bf9c1662992a492210e11af477cd3dae879c89a979c67f6821c9af1b3ec8abde.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gsettings-desktop-schemas/gsettings-desktop-schemas_3.36.0-1ubuntu1_all.deb"],
        deps = ["@ubuntu2004_dconf-gsettings-backend_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_gtk-update-icon-cache_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ee2035b43b03bb0db7c81c438a7a6354520116faded843cccd2b131c6f10be37",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ee2035b43b03bb0db7c81c438a7a6354520116faded843cccd2b131c6f10be37.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gtk+3.0/gtk-update-icon-cache_3.24.18-1ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libgdk-pixbuf2.0-0_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_hicolor-icon-theme_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9000eb98868252261978ff49501c0ace3124cb369c395e9f5015ddc556fe2ba6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9000eb98868252261978ff49501c0ace3124cb369c395e9f5015ddc556fe2ba6.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/h/hicolor-icon-theme/hicolor-icon-theme_0.17-2_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_humanity-icon-theme_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "861b3767d7d2e5b472bc9d001dd959be976b92e61cead0879ccb03fe4a2bde40",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/861b3767d7d2e5b472bc9d001dd959be976b92e61cead0879ccb03fe4a2bde40.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/h/humanity-icon-theme/humanity-icon-theme_0.6.15_all.deb"],
        deps = ["@ubuntu2004_hicolor-icon-theme_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_icu-devtools_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ef3a4e07a63574fcec04b721d0be4bc4f2fa606fcc18a76d7e585ec4938abf4b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ef3a4e07a63574fcec04b721d0be4bc4f2fa606fcc18a76d7e585ec4938abf4b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/i/icu/icu-devtools_66.1-2ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_libgcc-s1_aarch64//:all_files", "@ubuntu2004_libicu66_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_iso-codes_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d23ceffbfc1608a3ecc7a65495a6355bd8993a72f485909ca202762967e87b64",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d23ceffbfc1608a3ecc7a65495a6355bd8993a72f485909ca202762967e87b64.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/i/iso-codes/iso-codes_4.4-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libacl1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "218a0cc187a63c95ae3efb886e200f5dab1a9a3600e0e3781953c08acc61eff2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/218a0cc187a63c95ae3efb886e200f5dab1a9a3600e0e3781953c08acc61eff2.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/acl/libacl1_2.2.53-6_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libapparmor1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ede2b05224e8853b0311231a9daf385b47f3ef8b65068e0a87741a6275bff68a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ede2b05224e8853b0311231a9daf385b47f3ef8b65068e0a87741a6275bff68a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/apparmor/libapparmor1_2.13.3-7ubuntu5_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libasan5_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f418420988acfe9afc22bbd2e3ce9b1fcec718af08aef92299badc64b19c6583",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f418420988acfe9afc22bbd2e3ce9b1fcec718af08aef92299badc64b19c6583.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-9/libasan5_9.3.0-10ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_gcc-9-base_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libasound2-data_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ba0b40483f57adc8ebbfbf7a82d9d3f370c1fad2be8d219667f2ef5774baaa52",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ba0b40483f57adc8ebbfbf7a82d9d3f370c1fad2be8d219667f2ef5774baaa52.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/alsa-lib/libasound2-data_1.2.2-2.1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libasound2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f3c1b0e0f3148f434569f9d4c413f41a18c8fd75d0b7f4da356543988c1a304e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f3c1b0e0f3148f434569f9d4c413f41a18c8fd75d0b7f4da356543988c1a304e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/alsa-lib/libasound2_1.2.2-2.1_arm64.deb"],
        deps = ["@ubuntu2004_libasound2-data_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libatk-bridge2.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "406e350befb5379ed1363a1ad31d8347fc8518d43dba137da02e000dc284f9ac",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/406e350befb5379ed1363a1ad31d8347fc8518d43dba137da02e000dc284f9ac.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/at-spi2-atk/libatk-bridge2.0-0_2.34.1-3_arm64.deb"],
        deps = ["@ubuntu2004_libatk1.0-0_aarch64//:all_files", "@ubuntu2004_libatspi2.0-0_aarch64//:all_files", "@ubuntu2004_libdbus-1-3_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libatk1.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c67756f42f3655a896300145ad5faa313426e5e2353005e3a555ebad78fbe32e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c67756f42f3655a896300145ad5faa313426e5e2353005e3a555ebad78fbe32e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/atk1.0/libatk1.0-0_2.35.1-1ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_libatk1.0-data_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libatk1.0-data_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ded987a2b50f0fa9bfd04fba60ad817cb169fb4066f7bc22c174fcae1e2f1aad",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ded987a2b50f0fa9bfd04fba60ad817cb169fb4066f7bc22c174fcae1e2f1aad.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/atk1.0/libatk1.0-data_2.35.1-1ubuntu2_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libatomic1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2e22ed242782f65f8045073a6ff6a3c9a128751ae45ed71b5a96ff54421263d3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2e22ed242782f65f8045073a6ff6a3c9a128751ae45ed71b5a96ff54421263d3.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/libatomic1_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libatspi2.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7f77390e8cde88006cf4277ecf2e678b486f3032ac476a2ae261cf7fa7a9befc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7f77390e8cde88006cf4277ecf2e678b486f3032ac476a2ae261cf7fa7a9befc.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/at-spi2-core/libatspi2.0-0_2.36.0-2_arm64.deb"],
        deps = ["@ubuntu2004_libdbus-1-3_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libattr1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "5ef640b8b2c4824eaf564a07f1c39f68ec31334cda474507f427aeb55ea8d183",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5ef640b8b2c4824eaf564a07f1c39f68ec31334cda474507f427aeb55ea8d183.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/attr/libattr1_2.4.48-5_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libaudit-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9fa4a291df5682f5fee12aacdddd9ed09445c352c80d8a1af36056866e4b4906",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9fa4a291df5682f5fee12aacdddd9ed09445c352c80d8a1af36056866e4b4906.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/audit/libaudit-common_2.8.5-2ubuntu6_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libaudit1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c9bf5af721e15f208e27bd04e5b01826f136a7f0dcba5fe4eb06b1ea09abddff",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c9bf5af721e15f208e27bd04e5b01826f136a7f0dcba5fe4eb06b1ea09abddff.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/audit/libaudit1_2.8.5-2ubuntu6_arm64.deb"],
        deps = ["@ubuntu2004_libaudit-common_aarch64//:all_files", "@ubuntu2004_libcap-ng0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libavahi-client3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "61f0fc32f65b5795d324a8d0b8ec1ed8219ca6c4a0059f1bae04463b7ee65a06",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/61f0fc32f65b5795d324a8d0b8ec1ed8219ca6c4a0059f1bae04463b7ee65a06.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/avahi/libavahi-client3_0.7-4ubuntu7_arm64.deb"],
        deps = ["@ubuntu2004_libavahi-common3_aarch64//:all_files", "@ubuntu2004_libdbus-1-3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libavahi-common-data_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7c536e1b05cb1d5430709b826ee496a8e29d70078741de42663c41d698d549d6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7c536e1b05cb1d5430709b826ee496a8e29d70078741de42663c41d698d549d6.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/avahi/libavahi-common-data_0.7-4ubuntu7_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libavahi-common3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "439445394897b9bc91404ce31f1fa2394940408b8ae10f4e48bdd48269aad4e0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/439445394897b9bc91404ce31f1fa2394940408b8ae10f4e48bdd48269aad4e0.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/a/avahi/libavahi-common3_0.7-4ubuntu7_arm64.deb"],
        deps = ["@ubuntu2004_libavahi-common-data_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libbinutils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "423c33ccf87d155199d585b28818133bf9a1c9ab2243cc8bfa3dbc660727a600",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/423c33ccf87d155199d585b28818133bf9a1c9ab2243cc8bfa3dbc660727a600.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/binutils/libbinutils_2.34-6ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libblkid1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "14557fdbddbb9e36f4287cb42799c0e1d7cfb391872d14c8691d7450bc3dac4b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/14557fdbddbb9e36f4287cb42799c0e1d7cfb391872d14c8691d7450bc3dac4b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/u/util-linux/libblkid1_2.34-0.1ubuntu9_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libbrotli1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0cdec300b59d3a4ac2a39cc9dd493b2e301d050ef421fc315298f73c992b4e20",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0cdec300b59d3a4ac2a39cc9dd493b2e301d050ef421fc315298f73c992b4e20.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/brotli/libbrotli1_1.0.7-6build1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libbsd0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0486962198502bfd808a4645d453d3d9ee0916c29c4217355b61a3d206d77f4a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0486962198502bfd808a4645d453d3d9ee0916c29c4217355b61a3d206d77f4a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libb/libbsd/libbsd0_0.10.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libbz2-1.0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "6223fe0025beac129e9168be5a5ffc3655c23defd12de86b2f5dc52c3fd8d93b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6223fe0025beac129e9168be5a5ffc3655c23defd12de86b2f5dc52c3fd8d93b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/bzip2/libbz2-1.0_1.0.8-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libc-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8d105385d33f3f276d2ce28b7347b092f06aa4e8f30bf51045fb605825997d45",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8d105385d33f3f276d2ce28b7347b092f06aa4e8f30bf51045fb605825997d45.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/glibc/libc-bin_2.31-0ubuntu9_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libc-dev-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c169129b90b6b58e87a2fb006902cbde385bc971692db6e98edcd841f2bcc6e2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c169129b90b6b58e87a2fb006902cbde385bc971692db6e98edcd841f2bcc6e2.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/glibc/libc-dev-bin_2.31-0ubuntu9_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libc6-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7396b98f74412c7160c89717c51e50392c7d66c8d66fd7e6795cee2545feddce",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7396b98f74412c7160c89717c51e50392c7d66c8d66fd7e6795cee2545feddce.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/glibc/libc6-dev_2.31-0ubuntu9_arm64.deb"],
        deps = ["@ubuntu2004_libc-dev-bin_aarch64//:all_files", "@ubuntu2004_libcrypt-dev_aarch64//:all_files", "@ubuntu2004_linux-libc-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libc6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "783130400addeedde4c32e020d8e6bb612d9c8296eea66c74f2f86cbc938c033",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/783130400addeedde4c32e020d8e6bb612d9c8296eea66c74f2f86cbc938c033.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/glibc/libc6_2.31-0ubuntu9_arm64.deb"],
        deps = ["@ubuntu2004_libcrypt1_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcairo-gobject2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d2fa669102889605779cfdfc80f773fb463fe4253ef387ddb12cf37d18dad785",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d2fa669102889605779cfdfc80f773fb463fe4253ef387ddb12cf37d18dad785.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/c/cairo/libcairo-gobject2_1.16.0-4ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libcairo2_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcairo2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "afdf13c7d5ee20e910349a555d23d86ab373450e9eab099fe0cc1ddcbf410fba",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/afdf13c7d5ee20e910349a555d23d86ab373450e9eab099fe0cc1ddcbf410fba.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/c/cairo/libcairo2_1.16.0-4ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libfontconfig1_aarch64//:all_files", "@ubuntu2004_libfreetype6_aarch64//:all_files", "@ubuntu2004_libpixman-1-0_aarch64//:all_files", "@ubuntu2004_libpng16-16_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxcb-render0_aarch64//:all_files", "@ubuntu2004_libxcb-shm0_aarch64//:all_files", "@ubuntu2004_libxcb1_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files", "@ubuntu2004_libxrender1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcap-ng0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "5a362a556902427435b8e2c28d319263fa9a9c14848a7c7996da843d1ce45ec6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5a362a556902427435b8e2c28d319263fa9a9c14848a7c7996da843d1ce45ec6.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libc/libcap-ng/libcap-ng0_0.7.9-2.1build1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcap2-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a9e552514de8070c52f255c6a24c23baea0b7a92794ba3679c0d72555afddafd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a9e552514de8070c52f255c6a24c23baea0b7a92794ba3679c0d72555afddafd.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libc/libcap2/libcap2-bin_2.32-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcap2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0699b5c853c07148b92e610366573eb064cbffc2e7c7a0b39cd7dd6cdf4645f6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0699b5c853c07148b92e610366573eb064cbffc2e7c7a0b39cd7dd6cdf4645f6.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libc/libcap2/libcap2_2.32-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcc1-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "203835559db067a1cf4d2fb677d7fd888c44db6947e32d6a21d18a42b911b6b1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/203835559db067a1cf4d2fb677d7fd888c44db6947e32d6a21d18a42b911b6b1.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/libcc1-0_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcolord2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "efa0ec346faf21cfb794f6e79a461aafaf425d4d10e9fe37afda8201da22e880",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/efa0ec346faf21cfb794f6e79a461aafaf425d4d10e9fe37afda8201da22e880.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/c/colord/libcolord2_1.4.4-2_arm64.deb"],
        deps = ["@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_liblcms2-2_aarch64//:all_files", "@ubuntu2004_libudev1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcom-err2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c3799b2138619a50a6ff5fdb7e606577fa5111a3549bd07a6f80289a62a68aa1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c3799b2138619a50a6ff5fdb7e606577fa5111a3549bd07a6f80289a62a68aa1.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/e/e2fsprogs/libcom-err2_1.45.5-2ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcrypt-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3cbd65247c73ec78e15e9364d3f15643591fe25849382251c77a49204f083e68",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3cbd65247c73ec78e15e9364d3f15643591fe25849382251c77a49204f083e68.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcrypt/libcrypt-dev_4.4.10-10ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_libcrypt1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcrypt1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "89907b6c1b613c430a53ef10909934b9ce5854a396cd173360495f6f8e5e7ea4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/89907b6c1b613c430a53ef10909934b9ce5854a396cd173360495f6f8e5e7ea4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcrypt/libcrypt1_4.4.10-10ubuntu4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libctf-nobfd0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ae9545c4cbf921bbccdf3c958b92890a2263c6c759d538cbb1f312f861207b97",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ae9545c4cbf921bbccdf3c958b92890a2263c6c759d538cbb1f312f861207b97.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/binutils/libctf-nobfd0_2.34-6ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libctf0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f407c88ca15cea583b2c87dd2be19c51353f2dd04f0361888c797ffe9635fe42",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f407c88ca15cea583b2c87dd2be19c51353f2dd04f0361888c797ffe9635fe42.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/b/binutils/libctf0_2.34-6ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libcups2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "6bb0eed54b143be37742d00085e98a6c13751a42217dcd77aacf63009a67e2d4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6bb0eed54b143be37742d00085e98a6c13751a42217dcd77aacf63009a67e2d4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/c/cups/libcups2_2.3.1-9ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libavahi-client3_aarch64//:all_files", "@ubuntu2004_libavahi-common3_aarch64//:all_files", "@ubuntu2004_libgnutls30_aarch64//:all_files", "@ubuntu2004_libgssapi-krb5-2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdatrie1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "31918038b7125be45b4f94772a2305009487812556f0cf9002b4e1ec0e6965ca",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/31918038b7125be45b4f94772a2305009487812556f0cf9002b4e1ec0e6965ca.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdatrie/libdatrie1_0.2.12-3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdb5.3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c7559b42a507da72f2dcca86d70acca1349407416ea4c485b6fb36335879642a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c7559b42a507da72f2dcca86d70acca1349407416ea4c485b6fb36335879642a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/db5.3/libdb5.3_5.3.28+dfsg1-0.6ubuntu2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdbus-1-3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b04b9461049d1bcfa9b2455d6218d0cdf63414ae17612d8c6cfb8cacaf4ae70b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b04b9461049d1bcfa9b2455d6218d0cdf63414ae17612d8c6cfb8cacaf4ae70b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/dbus/libdbus-1-3_1.12.16-2ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_libsystemd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdconf1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f31974250346ff5b793acdb1455153103936ef4dac016a33f59506686a2a5610",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f31974250346ff5b793acdb1455153103936ef4dac016a33f59506686a2a5610.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/dconf/libdconf1_0.36.0-1_arm64.deb"],
        deps = ["@ubuntu2004_libglib2.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdpkg-perl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b7f34bb3574e2fae9054649d87a69e8d28a0f2c6a616d07a4f294d5462bb161c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b7f34bb3574e2fae9054649d87a69e8d28a0f2c6a616d07a4f294d5462bb161c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/d/dpkg/libdpkg-perl_1.19.7ubuntu3_all.deb"],
        deps = ["@ubuntu2004_perl_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdrm-amdgpu1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b06ae4a1ba3f49f31e8944135edb3734e30ea83e99ac450fd77e1d0d46e1054e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b06ae4a1ba3f49f31e8944135edb3734e30ea83e99ac450fd77e1d0d46e1054e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdrm/libdrm-amdgpu1_2.4.101-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdrm-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "fb5e29849ff773f15ecbfbe73634dd086292bd03607fde16c5cc4ed4449c9ccb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fb5e29849ff773f15ecbfbe73634dd086292bd03607fde16c5cc4ed4449c9ccb.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdrm/libdrm-common_2.4.101-2_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdrm-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "abec9887b23c4aae6b2477260d90086ac6b79335eaa64f96fc1b0f70b6c26a32",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/abec9887b23c4aae6b2477260d90086ac6b79335eaa64f96fc1b0f70b6c26a32.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdrm/libdrm-dev_2.4.101-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm-amdgpu1_aarch64//:all_files", "@ubuntu2004_libdrm-etnaviv1_aarch64//:all_files", "@ubuntu2004_libdrm-freedreno1_aarch64//:all_files", "@ubuntu2004_libdrm-nouveau2_aarch64//:all_files", "@ubuntu2004_libdrm-radeon1_aarch64//:all_files", "@ubuntu2004_libdrm-tegra0_aarch64//:all_files", "@ubuntu2004_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdrm-etnaviv1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "cdeae0dd9702b7485eb80309d5ee28488231d3eb7559921a6275e2b6c7cd6520",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cdeae0dd9702b7485eb80309d5ee28488231d3eb7559921a6275e2b6c7cd6520.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdrm/libdrm-etnaviv1_2.4.101-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdrm-freedreno1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c0af9ddaf1dae9e23867462d1df0412b72ae0f97a2538bc0a8b4bff4e1342f51",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c0af9ddaf1dae9e23867462d1df0412b72ae0f97a2538bc0a8b4bff4e1342f51.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdrm/libdrm-freedreno1_2.4.101-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdrm-nouveau2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ba4f06105eef16cf56da7b1103c970dfb3204f039e956aaedd1c86666f18884a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ba4f06105eef16cf56da7b1103c970dfb3204f039e956aaedd1c86666f18884a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdrm/libdrm-nouveau2_2.4.101-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdrm-radeon1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "fb049b79baed524c81393ea343a10064be8b41fc90e484e4e909e48458a0549a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fb049b79baed524c81393ea343a10064be8b41fc90e484e4e909e48458a0549a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdrm/libdrm-radeon1_2.4.101-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdrm-tegra0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7cc00781fb4d4b6ababfc0b6a8159c2d8dd5c5d0808c05b0f80d52ce555db40c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7cc00781fb4d4b6ababfc0b6a8159c2d8dd5c5d0808c05b0f80d52ce555db40c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdrm/libdrm-tegra0_2.4.101-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libdrm2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "01d2609dbe4baa07194e6f28b659f54e827f82b9923446be931bd78b8bf7e4e0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/01d2609dbe4baa07194e6f28b659f54e827f82b9923446be931bd78b8bf7e4e0.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libd/libdrm/libdrm2_2.4.101-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm-common_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libegl-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2d26b8e3d085792b9fc588a89cef229b129adbfe11ab4100c4367cb7177c430a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2d26b8e3d085792b9fc588a89cef229b129adbfe11ab4100c4367cb7177c430a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libegl-dev_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libegl1_aarch64//:all_files", "@ubuntu2004_libgl-dev_aarch64//:all_files", "@ubuntu2004_libx11-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libegl-mesa0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0febc320b2b9bc18061512167356f9a698f8af705411eb06ba0520b5c9186406",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0febc320b2b9bc18061512167356f9a698f8af705411eb06ba0520b5c9186406.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/mesa/libegl-mesa0_20.0.4-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files", "@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libgbm1_aarch64//:all_files", "@ubuntu2004_libglapi-mesa_aarch64//:all_files", "@ubuntu2004_libwayland-client0_aarch64//:all_files", "@ubuntu2004_libwayland-server0_aarch64//:all_files", "@ubuntu2004_libx11-xcb1_aarch64//:all_files", "@ubuntu2004_libxcb-dri2-0_aarch64//:all_files", "@ubuntu2004_libxcb-dri3-0_aarch64//:all_files", "@ubuntu2004_libxcb-present0_aarch64//:all_files", "@ubuntu2004_libxcb-sync1_aarch64//:all_files", "@ubuntu2004_libxcb-xfixes0_aarch64//:all_files", "@ubuntu2004_libxcb1_aarch64//:all_files", "@ubuntu2004_libxshmfence1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libegl1-mesa-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0f52993c7f8b421b9749841d93aebafb775401ab055aac158d6dc9dec22641e7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0f52993c7f8b421b9749841d93aebafb775401ab055aac158d6dc9dec22641e7.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/mesa/libegl1-mesa-dev_20.0.4-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libegl-dev_aarch64//:all_files", "@ubuntu2004_libglvnd-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libegl1-mesa_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e566a318dc118eb1a5dcb1af7af5cbdcebd2d98016774fbd5765eb9f582887bd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e566a318dc118eb1a5dcb1af7af5cbdcebd2d98016774fbd5765eb9f582887bd.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/universe/m/mesa/libegl1-mesa_20.0.4-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libegl1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libegl1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "1241af3bf19f4d963f46b655f0d7663cbfe3c7aba9369fa2cd136c978573b52d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1241af3bf19f4d963f46b655f0d7663cbfe3c7aba9369fa2cd136c978573b52d.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libegl1_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libegl-mesa0_aarch64//:all_files", "@ubuntu2004_libglvnd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libelf-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7474d0569f5811ebf141d84917c7bc9e177ed9c6a05a4dd63a8d7d9d3795c6e5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7474d0569f5811ebf141d84917c7bc9e177ed9c6a05a4dd63a8d7d9d3795c6e5.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/e/elfutils/libelf-dev_0.176-1.1build1_arm64.deb"],
        deps = ["@ubuntu2004_zlib1g-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libelf1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9085845162360fd2afb7a16ced944bf7383aaaf1c8687f15df135afc6451ab86",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9085845162360fd2afb7a16ced944bf7383aaaf1c8687f15df135afc6451ab86.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/e/elfutils/libelf1_0.176-1.1build1_arm64.deb"],
        deps = ["@ubuntu2004_zlib1g_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libepoxy0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c9b0eaf81fb298fcf54fe61591f316b674fb01757c889017aa86df7f8cf0c3a8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c9b0eaf81fb298fcf54fe61591f316b674fb01757c889017aa86df7f8cf0c3a8.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libe/libepoxy/libepoxy0_1.5.4-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libexpat1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d1b5c8750b1ba09cfe97e6dea4afb5ad73bb464d89544d790892562b49d9de13",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d1b5c8750b1ba09cfe97e6dea4afb5ad73bb464d89544d790892562b49d9de13.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/e/expat/libexpat1_2.2.9-1build1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libffi7_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a3e7abc48b5b4614fd5f6d1d95019d52be092e40e9b8a05f670905c6abf6eac4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a3e7abc48b5b4614fd5f6d1d95019d52be092e40e9b8a05f670905c6abf6eac4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libf/libffi/libffi7_3.3-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libfontconfig1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7abc3c68569ebd52e7e27ba1cd1e97004bf5a1641212acd821220a020b713c2f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7abc3c68569ebd52e7e27ba1cd1e97004bf5a1641212acd821220a020b713c2f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/f/fontconfig/libfontconfig1_2.13.1-2ubuntu3_arm64.deb"],
        deps = ["@ubuntu2004_fontconfig-config_aarch64//:all_files", "@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libfreetype6_aarch64//:all_files", "@ubuntu2004_libuuid1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libfreetype6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b573d29ea68eb053c8b3cc866dffd433e59a37f57679094480d43dd5ee7c24c9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b573d29ea68eb053c8b3cc866dffd433e59a37f57679094480d43dd5ee7c24c9.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/f/freetype/libfreetype6_2.10.1-2_arm64.deb"],
        deps = ["@ubuntu2004_libpng16-16_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libfribidi0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2b8320734acc65a60a5a976a9ae606be34765c171055948014978f3c4ba8cb24",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2b8320734acc65a60a5a976a9ae606be34765c171055948014978f3c4ba8cb24.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/f/fribidi/libfribidi0_1.0.8-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgbm1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f7ae679229d69969149e2e2c8b1eab0bc1f3d0d4557bd5138e66d0ea34839cc4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f7ae679229d69969149e2e2c8b1eab0bc1f3d0d4557bd5138e66d0ea34839cc4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/mesa/libgbm1_20.0.4-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files", "@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libwayland-server0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgcc-9-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2b9484358d26eb2a83c92acfe445965918f049e540c285e16b286f5be79dd435",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2b9484358d26eb2a83c92acfe445965918f049e540c285e16b286f5be79dd435.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-9/libgcc-9-dev_9.3.0-10ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_gcc-9-base_aarch64//:all_files", "@ubuntu2004_libasan5_aarch64//:all_files", "@ubuntu2004_libatomic1_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files", "@ubuntu2004_libgomp1_aarch64//:all_files", "@ubuntu2004_libitm1_aarch64//:all_files", "@ubuntu2004_liblsan0_aarch64//:all_files", "@ubuntu2004_libtsan0_aarch64//:all_files", "@ubuntu2004_libubsan1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgcc-s1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ac157ece7e0d41abf7c05309d8c85742f5f895a09c573add08152198390840ee",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ac157ece7e0d41abf7c05309d8c85742f5f895a09c573add08152198390840ee.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/libgcc-s1_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgcc1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2409e287b6f2b8d5f160f39603dc1f63d7c3a17ca4d48b29a65d495a6f12b555",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2409e287b6f2b8d5f160f39603dc1f63d7c3a17ca4d48b29a65d495a6f12b555.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/universe/g/gcc-10/libgcc1_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgcrypt20_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "60693d62488da3e5743d2fdd460c34555fbc487c948d5d08d96b1892ae941bd2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/60693d62488da3e5743d2fdd460c34555fbc487c948d5d08d96b1892ae941bd2.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libgcrypt20/libgcrypt20_1.8.5-5ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libgpg-error0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgdbm-compat4_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "41bf13d7e9af05fe8ba430717539ad211209be387f1d4c7bfa8e2d197295b2b3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/41bf13d7e9af05fe8ba430717539ad211209be387f1d4c7bfa8e2d197295b2b3.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gdbm/libgdbm-compat4_1.18.1-5_arm64.deb"],
        deps = ["@ubuntu2004_libgdbm6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgdbm6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3fe57483eae460224242495d7438beb12b32483a6481c8e4a1d36704f90e64aa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3fe57483eae460224242495d7438beb12b32483a6481c8e4a1d36704f90e64aa.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gdbm/libgdbm6_1.18.1-5_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgdk-pixbuf2.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b3d10f0565221b4f8428ac7e4e420c50dfdcb71baf333f79d5d489fc230aa432",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b3d10f0565221b4f8428ac7e4e420c50dfdcb71baf333f79d5d489fc230aa432.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gdk-pixbuf/libgdk-pixbuf2.0-0_2.40.0+dfsg-3_arm64.deb"],
        deps = ["@ubuntu2004_libgdk-pixbuf2.0-common_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libjpeg8_aarch64//:all_files", "@ubuntu2004_libpng16-16_aarch64//:all_files", "@ubuntu2004_libtiff5_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgdk-pixbuf2.0-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "247957ead285e60efcb766b35bdaf85425319f53555045090d9f1d5a9e49a0ec",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/247957ead285e60efcb766b35bdaf85425319f53555045090d9f1d5a9e49a0ec.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gdk-pixbuf/libgdk-pixbuf2.0-common_2.40.0+dfsg-3_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgl-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a6663cfb40ebb9ec76f3576640cf064d3681396ea75ecd622eb1519c7852f3dd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a6663cfb40ebb9ec76f3576640cf064d3681396ea75ecd622eb1519c7852f3dd.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libgl-dev_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libgl1_aarch64//:all_files", "@ubuntu2004_libglx-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgl1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "1b04e0b63bfbb8da77087cf2de8171ccf74d7baae4b128029ef938dc130f3bc4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1b04e0b63bfbb8da77087cf2de8171ccf74d7baae4b128029ef938dc130f3bc4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libgl1_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libglvnd0_aarch64//:all_files", "@ubuntu2004_libglx0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libglapi-mesa_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c4f3ba5c5c695c10ca436639e28985b798526a1a7ef45095392cfa779e729316",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c4f3ba5c5c695c10ca436639e28985b798526a1a7ef45095392cfa779e729316.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/mesa/libglapi-mesa_20.0.4-2ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgles-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "5ec50add6c92714aaa5509dd77df3a2a6220557d3bf0bea7704a544595f8286f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5ec50add6c92714aaa5509dd77df3a2a6220557d3bf0bea7704a544595f8286f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libgles-dev_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libgles1_aarch64//:all_files", "@ubuntu2004_libgles2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgles1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f9d535fbd49a58a3bf360550b627f21e325d16590ab73d7312b68b27d3a9c8af",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f9d535fbd49a58a3bf360550b627f21e325d16590ab73d7312b68b27d3a9c8af.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libgles1_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libglvnd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgles2-mesa-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7b61de1601ccb6e1dea99297ca87acff9d3d7862770b6bd32526f3231fcc8bf4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7b61de1601ccb6e1dea99297ca87acff9d3d7862770b6bd32526f3231fcc8bf4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/mesa/libgles2-mesa-dev_20.0.4-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libgles-dev_aarch64//:all_files", "@ubuntu2004_libglvnd-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgles2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "472e98965cd8ccc8536f2a7e6f7fcae0c67056e1018ce206bff8361ca4657464",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/472e98965cd8ccc8536f2a7e6f7fcae0c67056e1018ce206bff8361ca4657464.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libgles2_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libglvnd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libglib2.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "90788fed3382d5089bc4fe467d49a93f639a80ae9c7b3e6ae1a56f778aa9789b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/90788fed3382d5089bc4fe467d49a93f639a80ae9c7b3e6ae1a56f778aa9789b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/glib2.0/libglib2.0-0_2.64.2-1~fakesync1_arm64.deb"],
        deps = ["@ubuntu2004_libffi7_aarch64//:all_files", "@ubuntu2004_libmount1_aarch64//:all_files", "@ubuntu2004_libpcre3_aarch64//:all_files", "@ubuntu2004_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libglvnd-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2af37d931a49c69d3b03c503f1268df8725e9af634accab8c3d36f7b9e4f6570",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2af37d931a49c69d3b03c503f1268df8725e9af634accab8c3d36f7b9e4f6570.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libglvnd-dev_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libegl-dev_aarch64//:all_files", "@ubuntu2004_libgl-dev_aarch64//:all_files", "@ubuntu2004_libgles-dev_aarch64//:all_files", "@ubuntu2004_libglvnd0_aarch64//:all_files", "@ubuntu2004_libglx-dev_aarch64//:all_files", "@ubuntu2004_libopengl-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libglvnd0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "eef82b1ad201624beef7ba849069345adcc35e95a83afdd07198c96921f5ff7a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/eef82b1ad201624beef7ba849069345adcc35e95a83afdd07198c96921f5ff7a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libglvnd0_1.3.1-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libglx-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8da0c32ab82309449b5386176236f2e20079f76b59c95ee00cf8a20836d81ad0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8da0c32ab82309449b5386176236f2e20079f76b59c95ee00cf8a20836d81ad0.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libglx-dev_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libglx0_aarch64//:all_files", "@ubuntu2004_libx11-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libglx-mesa0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f1c097a25aa0e12fd60143958b07a5fe0f58bd3f00d3d934ad9aac01b39c73d3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f1c097a25aa0e12fd60143958b07a5fe0f58bd3f00d3d934ad9aac01b39c73d3.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/mesa/libglx-mesa0_20.0.4-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files", "@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libglapi-mesa_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libx11-xcb1_aarch64//:all_files", "@ubuntu2004_libxcb-dri2-0_aarch64//:all_files", "@ubuntu2004_libxcb-dri3-0_aarch64//:all_files", "@ubuntu2004_libxcb-glx0_aarch64//:all_files", "@ubuntu2004_libxcb-present0_aarch64//:all_files", "@ubuntu2004_libxcb-sync1_aarch64//:all_files", "@ubuntu2004_libxcb1_aarch64//:all_files", "@ubuntu2004_libxdamage1_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files", "@ubuntu2004_libxfixes3_aarch64//:all_files", "@ubuntu2004_libxshmfence1_aarch64//:all_files", "@ubuntu2004_libxxf86vm1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libglx0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "bdeb9deae03be71ae87f45e3f843c9ff197a39d2995d3da5a843ef863ddfd60d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bdeb9deae03be71ae87f45e3f843c9ff197a39d2995d3da5a843ef863ddfd60d.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libglx0_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libglx-mesa0_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgmp10_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "64d9fa3596b8b3f23626ffd0bd451b80fc27cfd3ea3cc64289640d9794946074",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/64d9fa3596b8b3f23626ffd0bd451b80fc27cfd3ea3cc64289640d9794946074.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gmp/libgmp10_6.2.0+dfsg-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgnutls30_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e13175bb1dc1620b6fdda31eb16b9aa19923af0447421212daf1475dfc35ae62",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e13175bb1dc1620b6fdda31eb16b9aa19923af0447421212daf1475dfc35ae62.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gnutls28/libgnutls30_3.6.13-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libgmp10_aarch64//:all_files", "@ubuntu2004_libhogweed5_aarch64//:all_files", "@ubuntu2004_libidn2-0_aarch64//:all_files", "@ubuntu2004_libnettle7_aarch64//:all_files", "@ubuntu2004_libp11-kit0_aarch64//:all_files", "@ubuntu2004_libtasn1-6_aarch64//:all_files", "@ubuntu2004_libunistring2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgomp1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "1a13c3f75acb395407fdd5f99032669395f7eb4fcaf0d330443cb1c295361bb4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/1a13c3f75acb395407fdd5f99032669395f7eb4fcaf0d330443cb1c295361bb4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/libgomp1_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgpg-error0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3b1364e9125ec484cfc3dd4441b836e111cf99e86274ca5e24a82d6c8343f15f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3b1364e9125ec484cfc3dd4441b836e111cf99e86274ca5e24a82d6c8343f15f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libgpg-error/libgpg-error0_1.37-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgraphite2-3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7eda3f4fa7eb67fbe9272ec52f73e09bd16ea0109817081e434119327a66f620",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7eda3f4fa7eb67fbe9272ec52f73e09bd16ea0109817081e434119327a66f620.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/graphite2/libgraphite2-3_1.3.13-11build1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgssapi-krb5-2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "be70a3cb969958b21efcb6a251a019e152cfc634dc1da5faae99825d748c228d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/be70a3cb969958b21efcb6a251a019e152cfc634dc1da5faae99825d748c228d.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/k/krb5/libgssapi-krb5-2_1.17-6ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_libcom-err2_aarch64//:all_files", "@ubuntu2004_libk5crypto3_aarch64//:all_files", "@ubuntu2004_libkrb5-3_aarch64//:all_files", "@ubuntu2004_libkrb5support0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgstreamer-plugins-base1.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8bdedfeb470fffbeef17f3c14b7146c52d4d63e0ca53ab6dce196faa8637325d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8bdedfeb470fffbeef17f3c14b7146c52d4d63e0ca53ab6dce196faa8637325d.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gst-plugins-base1.0/libgstreamer-plugins-base1.0-0_1.16.2-4_arm64.deb"],
        deps = ["@ubuntu2004_iso-codes_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_liborc-0.4-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgstreamer1.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "80bfe3fb98e295c776ce9601a1dffc876704474a2de59786642e24c0236951d5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/80bfe3fb98e295c776ce9601a1dffc876704474a2de59786642e24c0236951d5.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gstreamer1.0/libgstreamer1.0-0_1.16.2-2_arm64.deb"],
        deps = ["@ubuntu2004_libcap2-bin_aarch64//:all_files", "@ubuntu2004_libcap2_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgtk-3-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3bd8e8d84113d866fb8a70c33f7018d1ecc77ce8c8aae31b54567d6e396d61b9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3bd8e8d84113d866fb8a70c33f7018d1ecc77ce8c8aae31b54567d6e396d61b9.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gtk+3.0/libgtk-3-0_3.24.18-1ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_adwaita-icon-theme_aarch64//:all_files", "@ubuntu2004_hicolor-icon-theme_aarch64//:all_files", "@ubuntu2004_libatk-bridge2.0-0_aarch64//:all_files", "@ubuntu2004_libatk1.0-0_aarch64//:all_files", "@ubuntu2004_libcairo-gobject2_aarch64//:all_files", "@ubuntu2004_libcairo2_aarch64//:all_files", "@ubuntu2004_libcolord2_aarch64//:all_files", "@ubuntu2004_libcups2_aarch64//:all_files", "@ubuntu2004_libepoxy0_aarch64//:all_files", "@ubuntu2004_libfontconfig1_aarch64//:all_files", "@ubuntu2004_libfreetype6_aarch64//:all_files", "@ubuntu2004_libfribidi0_aarch64//:all_files", "@ubuntu2004_libgdk-pixbuf2.0-0_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libgtk-3-common_aarch64//:all_files", "@ubuntu2004_libharfbuzz0b_aarch64//:all_files", "@ubuntu2004_libjson-glib-1.0-0_aarch64//:all_files", "@ubuntu2004_libpango-1.0-0_aarch64//:all_files", "@ubuntu2004_libpangocairo-1.0-0_aarch64//:all_files", "@ubuntu2004_libpangoft2-1.0-0_aarch64//:all_files", "@ubuntu2004_librest-0.7-0_aarch64//:all_files", "@ubuntu2004_libwayland-client0_aarch64//:all_files", "@ubuntu2004_libwayland-cursor0_aarch64//:all_files", "@ubuntu2004_libwayland-egl1_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxcomposite1_aarch64//:all_files", "@ubuntu2004_libxcursor1_aarch64//:all_files", "@ubuntu2004_libxdamage1_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files", "@ubuntu2004_libxfixes3_aarch64//:all_files", "@ubuntu2004_libxi6_aarch64//:all_files", "@ubuntu2004_libxinerama1_aarch64//:all_files", "@ubuntu2004_libxkbcommon0_aarch64//:all_files", "@ubuntu2004_libxrandr2_aarch64//:all_files", "@ubuntu2004_shared-mime-info_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libgtk-3-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "dc21ce70514297460276011f9b2afeb2e1c7a4f2ee1b84a3554477c6bce8f29d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/dc21ce70514297460276011f9b2afeb2e1c7a4f2ee1b84a3554477c6bce8f29d.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gtk+3.0/libgtk-3-common_3.24.18-1ubuntu1_all.deb"],
        deps = ["@ubuntu2004_dconf-gsettings-backend_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libharfbuzz0b_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "61200d168f2388b7168e22e676ca9f11e27b316814c73cf8f17255eb30b342da",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/61200d168f2388b7168e22e676ca9f11e27b316814c73cf8f17255eb30b342da.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/h/harfbuzz/libharfbuzz0b_2.6.4-1ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_libfreetype6_aarch64//:all_files", "@ubuntu2004_libgraphite2-3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libhogweed5_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e6c25e4e0dc4a82b1f1c498b56677a5cbd265fee1eeda3a205787c37e85ab1ce",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e6c25e4e0dc4a82b1f1c498b56677a5cbd265fee1eeda3a205787c37e85ab1ce.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/n/nettle/libhogweed5_3.5.1+really3.5.1-2_arm64.deb"],
        deps = ["@ubuntu2004_libnettle7_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libicu-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "34db72d78908f3bdc4d75ed8ee848b8aee0950332f424b91fb6fb592e77eb764",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/34db72d78908f3bdc4d75ed8ee848b8aee0950332f424b91fb6fb592e77eb764.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/i/icu/libicu-dev_66.1-2ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_icu-devtools_aarch64//:all_files", "@ubuntu2004_libicu66_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libicu66_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "6302e309ab002af30ddfa0d68de26c68f7c034ed2f45b1d97a712bff1a03999a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6302e309ab002af30ddfa0d68de26c68f7c034ed2f45b1d97a712bff1a03999a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/i/icu/libicu66_66.1-2ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_libgcc-s1_aarch64//:all_files", "@ubuntu2004_tzdata_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libidn2-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "87b95a889659edc2cae07679a930880b625449e28608cb22a8b6b6e6149c22cd",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/87b95a889659edc2cae07679a930880b625449e28608cb22a8b6b6e6149c22cd.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libi/libidn2/libidn2-0_2.2.0-2_arm64.deb"],
        deps = ["@ubuntu2004_libunistring2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libisl22_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "60b341bc2988b17e6988592953730d0bb258a406da307fe9185253a3f48b851b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/60b341bc2988b17e6988592953730d0bb258a406da307fe9185253a3f48b851b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/i/isl/libisl22_0.22.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libgmp10_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libitm1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b28dd85519f6a5462ced2928e5c4ef33fba4adbfdd5bbe86df62a98e4e3688f3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b28dd85519f6a5462ced2928e5c4ef33fba4adbfdd5bbe86df62a98e4e3688f3.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/libitm1_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libjbig0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "43d66d6a8b545abb5ce57e621cc827be4aa06ad070942a31d2c20aafe1fb5a34",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/43d66d6a8b545abb5ce57e621cc827be4aa06ad070942a31d2c20aafe1fb5a34.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/j/jbigkit/libjbig0_2.1-3.1build1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libjpeg-turbo8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "cd6588ac233b9c1385ae6490e1969f15fe87b88b4f554dff0504eb4067edeb63",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/cd6588ac233b9c1385ae6490e1969f15fe87b88b4f554dff0504eb4067edeb63.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libj/libjpeg-turbo/libjpeg-turbo8_2.0.3-0ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libjpeg8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "89f4acf2407dc4f023b7b59c874557e4a94f67a9155ad5870b9ead1cb375cc13",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/89f4acf2407dc4f023b7b59c874557e4a94f67a9155ad5870b9ead1cb375cc13.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libj/libjpeg8-empty/libjpeg8_8c-2ubuntu8_arm64.deb"],
        deps = ["@ubuntu2004_libjpeg-turbo8_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libjson-glib-1.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d4de93bb2a2f5daaed11060fdc64007f84e64fc62952bdf8aba35b723db7d2f3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d4de93bb2a2f5daaed11060fdc64007f84e64fc62952bdf8aba35b723db7d2f3.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/j/json-glib/libjson-glib-1.0-0_1.4.4-2ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libjson-glib-1.0-common_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libjson-glib-1.0-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0e160489f656481cba2e62588dd367a880c866b46884fc183cfa965a33e692db",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0e160489f656481cba2e62588dd367a880c866b46884fc183cfa965a33e692db.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/j/json-glib/libjson-glib-1.0-common_1.4.4-2ubuntu2_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libk5crypto3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8ed8a19d27507973d07df7946dd6136ecce97080bf3da3cf272cc09e155b5a76",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8ed8a19d27507973d07df7946dd6136ecce97080bf3da3cf272cc09e155b5a76.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/k/krb5/libk5crypto3_1.17-6ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_libkrb5support0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libkeyutils1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e24a45844340d01b3ce5fd070654fbca61dd9e30ddd038e70b908424a880ad04",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e24a45844340d01b3ce5fd070654fbca61dd9e30ddd038e70b908424a880ad04.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/k/keyutils/libkeyutils1_1.6-6ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libkrb5-3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "27f45dff44d0c81d2505740092b007a57e910a8653f787e07c00eca98ebb0a1f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/27f45dff44d0c81d2505740092b007a57e910a8653f787e07c00eca98ebb0a1f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/k/krb5/libkrb5-3_1.17-6ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_libkeyutils1_aarch64//:all_files", "@ubuntu2004_libkrb5support0_aarch64//:all_files", "@ubuntu2004_libssl1.1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libkrb5support0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d86122e8cfeaa39a2bd99b1e420149a365934f3fe3b33bc2e354abd376650d1e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d86122e8cfeaa39a2bd99b1e420149a365934f3fe3b33bc2e354abd376650d1e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/k/krb5/libkrb5support0_1.17-6ubuntu4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_liblcms2-2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a829553cec6d8dfdc5fc22741aaedb180db878ff5de6cec246b97c1c7a2bfc9b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a829553cec6d8dfdc5fc22741aaedb180db878ff5de6cec246b97c1c7a2bfc9b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/l/lcms2/liblcms2-2_2.9-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_liblsan0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b4b52c81fb714d0dc72c1d959bf56c985003ddf1bfc3f1480601f2b18942c5af",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b4b52c81fb714d0dc72c1d959bf56c985003ddf1bfc3f1480601f2b18942c5af.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/liblsan0_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_liblz4-1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "302ebe6fd36a20e57f84eb2085c0e0ab5df2edc989871a037123d81e6fb90c4a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/302ebe6fd36a20e57f84eb2085c0e0ab5df2edc989871a037123d81e6fb90c4a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/l/lz4/liblz4-1_1.9.2-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_liblzma-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ba963e72637485339d5e894d89e6d850a022f6ddea430c0e5fc6778966aed8bb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ba963e72637485339d5e894d89e6d850a022f6ddea430c0e5fc6778966aed8bb.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/x/xz-utils/liblzma-dev_5.2.4-1_arm64.deb"],
        deps = ["@ubuntu2004_liblzma5_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_liblzma5_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "afba8c1c57b7aba8e8d8839e7af424044b7946862e59d4193c6af7d32637cd5c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/afba8c1c57b7aba8e8d8839e7af424044b7946862e59d4193c6af7d32637cd5c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/x/xz-utils/liblzma5_5.2.4-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libmount1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "317bc180a931d95f614206ef9742df6ec237ac9fd8f425d098a8d62eb075e0f2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/317bc180a931d95f614206ef9742df6ec237ac9fd8f425d098a8d62eb075e0f2.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/u/util-linux/libmount1_2.34-0.1ubuntu9_arm64.deb"],
        deps = ["@ubuntu2004_libblkid1_aarch64//:all_files", "@ubuntu2004_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libmpc3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "94dedb869bdc811aec4e8d5d794cd838e7d41765b5ffbdc5628eb57b6885bcb5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/94dedb869bdc811aec4e8d5d794cd838e7d41765b5ffbdc5628eb57b6885bcb5.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/mpclib3/libmpc3_1.1.0-1_arm64.deb"],
        deps = ["@ubuntu2004_libgmp10_aarch64//:all_files", "@ubuntu2004_libmpfr6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libmpfr6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "fa167728c18447512a3d680bf75c39f21d6a8336d7be5b9a60159cb86d95553c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fa167728c18447512a3d680bf75c39f21d6a8336d7be5b9a60159cb86d95553c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/mpfr4/libmpfr6_4.0.2-1_arm64.deb"],
        deps = ["@ubuntu2004_libgmp10_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libncurses-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "481b6b67ae45e21af7a6c0a2a12f63871f805d107446c5b22f408d180cb1343a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/481b6b67ae45e21af7a6c0a2a12f63871f805d107446c5b22f408d180cb1343a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/n/ncurses/libncurses-dev_6.2-0ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_libc6-dev_aarch64//:all_files", "@ubuntu2004_libncurses6_aarch64//:all_files", "@ubuntu2004_libncursesw6_aarch64//:all_files", "@ubuntu2004_ncurses-bin_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libncurses6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "83b26187127d9c4eee113d4cb2264c684d7e24e5adcdcfed6b15d55e134b0b74",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/83b26187127d9c4eee113d4cb2264c684d7e24e5adcdcfed6b15d55e134b0b74.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/n/ncurses/libncurses6_6.2-0ubuntu2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libncursesw6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7dfa81ae99d873cb3478ec577b5e20508e5b6965b36cd5a1b00495b3b93ad6ce",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7dfa81ae99d873cb3478ec577b5e20508e5b6965b36cd5a1b00495b3b93ad6ce.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/n/ncurses/libncursesw6_6.2-0ubuntu2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libnettle7_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9926a9b7920fa02a9828f6eb5d0e4e26f2c69684befe53582274473024276605",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9926a9b7920fa02a9828f6eb5d0e4e26f2c69684befe53582274473024276605.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/n/nettle/libnettle7_3.5.1+really3.5.1-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libnl-3-200_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8d576f3c5b590b4fa25c5e9831b00836bc2da6bd335d60ad371e523b75e9b315",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8d576f3c5b590b4fa25c5e9831b00836bc2da6bd335d60ad371e523b75e9b315.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libn/libnl3/libnl-3-200_3.4.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libnl-genl-3-200_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "07ff61942b2d5db844137c170108f73c4b03264c5dadf9c6f92f05506b3b18d4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/07ff61942b2d5db844137c170108f73c4b03264c5dadf9c6f92f05506b3b18d4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libn/libnl3/libnl-genl-3-200_3.4.0-1_arm64.deb"],
        deps = ["@ubuntu2004_libnl-3-200_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libnl-route-3-200_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b43e48f4c5655378d6f12f10140c70db8ff0b50094a1f75e4075e0e7a94ae958",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b43e48f4c5655378d6f12f10140c70db8ff0b50094a1f75e4075e0e7a94ae958.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libn/libnl3/libnl-route-3-200_3.4.0-1_arm64.deb"],
        deps = ["@ubuntu2004_libnl-3-200_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libopengl-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3f0365c31cbff892550291d42e1a9cf4760c9a83b906434df152cc0dd9343361",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3f0365c31cbff892550291d42e1a9cf4760c9a83b906434df152cc0dd9343361.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libopengl-dev_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libopengl0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libopengl0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a7afda3ca4a991aca2a593b60742cda572d119683ed3515776b26765a994e7be",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a7afda3ca4a991aca2a593b60742cda572d119683ed3515776b26765a994e7be.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libg/libglvnd/libopengl0_1.3.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libglvnd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_liborc-0.4-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "5b6c32b39217b0c917a6a84a1cc755687be54699410bc244370bd334f60285e1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/5b6c32b39217b0c917a6a84a1cc755687be54699410bc244370bd334f60285e1.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/o/orc/liborc-0.4-0_0.4.31-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libp11-kit0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3ba3cb8b5c263ee3a5419c69f2a6e5ff960def7a4a3325570f7d94cbaf7a99c4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3ba3cb8b5c263ee3a5419c69f2a6e5ff960def7a4a3325570f7d94cbaf7a99c4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/p11-kit/libp11-kit0_0.23.20-1build1_arm64.deb"],
        deps = ["@ubuntu2004_libffi7_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpam-modules-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f9dda4bfd63fec774289c3c615638f80de83c3013657c2b07975048b51e3ddfa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f9dda4bfd63fec774289c3c615638f80de83c3013657c2b07975048b51e3ddfa.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pam/libpam-modules-bin_1.3.1-5ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpam-modules_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "115e4201e7acb33eedc66d2c0a814c1eb93108a1c9869e13e48a8cc573849f26",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/115e4201e7acb33eedc66d2c0a814c1eb93108a1c9869e13e48a8cc573849f26.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pam/libpam-modules_1.3.1-5ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_debconf_aarch64//:all_files", "@ubuntu2004_libaudit1_aarch64//:all_files", "@ubuntu2004_libdb5.3_aarch64//:all_files", "@ubuntu2004_libpam-modules-bin_aarch64//:all_files", "@ubuntu2004_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpam-runtime_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "050de6a7503f260736fd511d4f609acc20e87e0309200b0950c505059dc5c01f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/050de6a7503f260736fd511d4f609acc20e87e0309200b0950c505059dc5c01f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pam/libpam-runtime_1.3.1-5ubuntu4_all.deb"],
        deps = ["@ubuntu2004_debconf_aarch64//:all_files", "@ubuntu2004_libpam-modules_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpam-systemd_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "56f5ab637b561d444d56c76f12b083691f873de2c0e99be310037228c9d4d924",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/56f5ab637b561d444d56c76f12b083691f873de2c0e99be310037228c9d4d924.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/s/systemd/libpam-systemd_245.4-4ubuntu3_arm64.deb"],
        deps = ["@ubuntu2004_libpam-runtime_aarch64//:all_files", "@ubuntu2004_libpam0g_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpam0g_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7c7fcb9765af337e075f3dc6cded9f8b6723c0e7068c61f6db73661332a97215",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7c7fcb9765af337e075f3dc6cded9f8b6723c0e7068c61f6db73661332a97215.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pam/libpam0g_1.3.1-5ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_debconf_aarch64//:all_files", "@ubuntu2004_libaudit1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpango-1.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a3f8dd6f8ff2bbb76d062673afc821d5251969dfc68454dce131b261fae55523",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a3f8dd6f8ff2bbb76d062673afc821d5251969dfc68454dce131b261fae55523.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pango1.0/libpango-1.0-0_1.44.7-2ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_fontconfig_aarch64//:all_files", "@ubuntu2004_libfribidi0_aarch64//:all_files", "@ubuntu2004_libthai0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpangocairo-1.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7b38d684e17acbcd77e1c231f32d96d19c6c62385b1d6098b2a0314f4488e0f0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7b38d684e17acbcd77e1c231f32d96d19c6c62385b1d6098b2a0314f4488e0f0.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pango1.0/libpangocairo-1.0-0_1.44.7-2ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_libcairo2_aarch64//:all_files", "@ubuntu2004_libfontconfig1_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libpango-1.0-0_aarch64//:all_files", "@ubuntu2004_libpangoft2-1.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpangoft2-1.0-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "22c073e49dbc31e3494d0b904a583a66d82e29bc7686e96357a7064a0f39bca4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/22c073e49dbc31e3494d0b904a583a66d82e29bc7686e96357a7064a0f39bca4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pango1.0/libpangoft2-1.0-0_1.44.7-2ubuntu4_arm64.deb"],
        deps = ["@ubuntu2004_libfreetype6_aarch64//:all_files", "@ubuntu2004_libpango-1.0-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpcre2-8-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "741444da6d973b2dcde1c700b0c85cd4a7d308dd016f226e80a007d10964b3de",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/741444da6d973b2dcde1c700b0c85cd4a7d308dd016f226e80a007d10964b3de.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pcre2/libpcre2-8-0_10.34-7_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpcre3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "588dc1cf9ee3d5a1b0eb918e8d023b9c97e6b4a8691daf62b32f31e8ba254e4f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/588dc1cf9ee3d5a1b0eb918e8d023b9c97e6b4a8691daf62b32f31e8ba254e4f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pcre3/libpcre3_8.39-12build1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libperl5.30_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "47d6aae207530997520b9954150d10ff9254b8f7be3eddc372b9f11434e67aff",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/47d6aae207530997520b9954150d10ff9254b8f7be3eddc372b9f11434e67aff.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/perl/libperl5.30_5.30.0-9build1_arm64.deb"],
        deps = ["@ubuntu2004_libbz2-1.0_aarch64//:all_files", "@ubuntu2004_libcrypt1_aarch64//:all_files", "@ubuntu2004_libdb5.3_aarch64//:all_files", "@ubuntu2004_libgdbm-compat4_aarch64//:all_files", "@ubuntu2004_libgdbm6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpixman-1-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "926a81b92f3e808a09f024a148b184dc68f61ea3a03dc26184c97f530fa5c04a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/926a81b92f3e808a09f024a148b184dc68f61ea3a03dc26184c97f530fa5c04a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/pixman/libpixman-1-0_0.38.4-0ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpng16-16_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "95185d067fcdee1fcc51cb83490720fd2dcee450c55b9cfabcc940ed160879b5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/95185d067fcdee1fcc51cb83490720fd2dcee450c55b9cfabcc940ed160879b5.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libp/libpng1.6/libpng16-16_1.6.37-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libproxy1v5_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7589bf5d3f5a801a3d0b28881c1e6c0c5ca9c243fa27864b8fbf70ab1d6f04aa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7589bf5d3f5a801a3d0b28881c1e6c0c5ca9c243fa27864b8fbf70ab1d6f04aa.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libp/libproxy/libproxy1v5_0.4.15-10_arm64.deb"],
        deps = ["@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpsl5_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d576cbdc18da58a33c955c76dfa709d6105b7bbd4bb9fd9ec1fda72309fe5764",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d576cbdc18da58a33c955c76dfa709d6105b7bbd4bb9fd9ec1fda72309fe5764.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libp/libpsl/libpsl5_0.21.0-1ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libidn2-0_aarch64//:all_files", "@ubuntu2004_libunistring2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libpthread-stubs0-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "23fe28d430b3b45b8c55fb147150f4303103a51a491f098959b117dc14971699",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/23fe28d430b3b45b8c55fb147150f4303103a51a491f098959b117dc14971699.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libp/libpthread-stubs/libpthread-stubs0-dev_0.4-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libreadline8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d017810a798b617f750519537726bdfcd5bca28d3c56598558322250d9012b45",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d017810a798b617f750519537726bdfcd5bca28d3c56598558322250d9012b45.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/r/readline/libreadline8_8.0-4_arm64.deb"],
        deps = ["@ubuntu2004_readline-common_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_librest-0.7-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "756ff8702af56cc0913d4db35d25f33e9030f3fa74a6f2a908ff589c36614bad",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/756ff8702af56cc0913d4db35d25f33e9030f3fa74a6f2a908ff589c36614bad.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libr/librest/librest-0.7-0_0.8.1-1_arm64.deb"],
        deps = ["@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libsoup-gnome2.4-1_aarch64//:all_files", "@ubuntu2004_libsoup2.4-1_aarch64//:all_files", "@ubuntu2004_libxml2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_librsvg2-2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4353ef1c56f95f1999cc756d1c2b656c8120d9feec9aef48672f8602db5d814b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4353ef1c56f95f1999cc756d1c2b656c8120d9feec9aef48672f8602db5d814b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libr/librsvg/librsvg2-2_2.48.2-1_arm64.deb"],
        deps = ["@ubuntu2004_libcairo2_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files", "@ubuntu2004_libgdk-pixbuf2.0-0_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libpango-1.0-0_aarch64//:all_files", "@ubuntu2004_libpangocairo-1.0-0_aarch64//:all_files", "@ubuntu2004_libxml2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_librsvg2-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "77994f1b82ef443e11b84957f88ac7a4066afed7ba998484ae085e79b8acb3ae",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/77994f1b82ef443e11b84957f88ac7a4066afed7ba998484ae085e79b8acb3ae.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libr/librsvg/librsvg2-common_2.48.2-1_arm64.deb"],
        deps = ["@ubuntu2004_libgdk-pixbuf2.0-0_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_librsvg2-2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libselinux1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "91006c0db207cb6df4205db9c5a91309db58e7f4eef649093aa89def780e0e66",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/91006c0db207cb6df4205db9c5a91309db58e7f4eef649093aa89def780e0e66.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libs/libselinux/libselinux1_3.0-1build2_arm64.deb"],
        deps = ["@ubuntu2004_libpcre2-8-0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libsemanage-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4141f803c811277d2ea56568a676a79f06017b8c5eb57891741808a27c55fffb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4141f803c811277d2ea56568a676a79f06017b8c5eb57891741808a27c55fffb.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libs/libsemanage/libsemanage-common_3.0-1build2_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libsemanage1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "6badb63e2650290230e42060c3461c5d30e1ed40dfdc8e1cfd80674cf22b76d0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6badb63e2650290230e42060c3461c5d30e1ed40dfdc8e1cfd80674cf22b76d0.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libs/libsemanage/libsemanage1_3.0-1build2_arm64.deb"],
        deps = ["@ubuntu2004_libaudit1_aarch64//:all_files", "@ubuntu2004_libbz2-1.0_aarch64//:all_files", "@ubuntu2004_libselinux1_aarch64//:all_files", "@ubuntu2004_libsemanage-common_aarch64//:all_files", "@ubuntu2004_libsepol1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libsepol1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e6e0a4fc9175f1dc8e3a90d0937975e25c834f20bb43999c411ca5a42a4268de",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e6e0a4fc9175f1dc8e3a90d0937975e25c834f20bb43999c411ca5a42a4268de.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libs/libsepol/libsepol1_3.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libset-scalar-perl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a398e400aedd0d19412fa61fed16b6d968926c4324322224c5b859d50dab63fa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a398e400aedd0d19412fa61fed16b6d968926c4324322224c5b859d50dab63fa.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/universe/libs/libset-scalar-perl/libset-scalar-perl_1.29-2_all.deb"],
        deps = ["@ubuntu2004_perl_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libsigsegv2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "821e75274e0d453008809eca1f9d422984b5f9be0d032cc4fc989395d02a08d2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/821e75274e0d453008809eca1f9d422984b5f9be0d032cc4fc989395d02a08d2.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libs/libsigsegv/libsigsegv2_2.12-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libsoup-gnome2.4-1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "98472df38789ad5b25042268ba6d17a416b315899ca721678b34b8705300811c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/98472df38789ad5b25042268ba6d17a416b315899ca721678b34b8705300811c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libs/libsoup2.4/libsoup-gnome2.4-1_2.70.0-1_arm64.deb"],
        deps = ["@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libsoup2.4-1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libsoup2.4-1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "7084f4dbf3ce883e9087b1851f8d3ea4d4b280a8198c82d9f314794f1e6947da",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/7084f4dbf3ce883e9087b1851f8d3ea4d4b280a8198c82d9f314794f1e6947da.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libs/libsoup2.4/libsoup2.4-1_2.70.0-1_arm64.deb"],
        deps = ["@ubuntu2004_glib-networking_aarch64//:all_files", "@ubuntu2004_libbrotli1_aarch64//:all_files", "@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libgssapi-krb5-2_aarch64//:all_files", "@ubuntu2004_libpsl5_aarch64//:all_files", "@ubuntu2004_libsqlite3-0_aarch64//:all_files", "@ubuntu2004_libxml2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libsqlite3-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2282ebd374efacea8d1627d6e76299939d75dcf83707e665b8d66b5b24889443",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2282ebd374efacea8d1627d6e76299939d75dcf83707e665b8d66b5b24889443.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/s/sqlite3/libsqlite3-0_3.31.1-4_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libssl1.1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a697e5826bdbed1324f3cce1335ef162bf49eed433eb662c6d43e69ebc4807b2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a697e5826bdbed1324f3cce1335ef162bf49eed433eb662c6d43e69ebc4807b2.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_debconf_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libstdc__-9-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "efbe47ef19a906dc0edd7031f0fabe332268adb7493b56587b30e181d5bef6b2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/efbe47ef19a906dc0edd7031f0fabe332268adb7493b56587b30e181d5bef6b2.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-9/libstdc++-9-dev_9.3.0-10ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_gcc-9-base_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libstdc__6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "bf29481e8373a4da958104daeaf32bb05fa405121a3aa0bbfbbd5d462725df3c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bf29481e8373a4da958104daeaf32bb05fa405121a3aa0bbfbbd5d462725df3c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/libstdc++6_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libsystemd0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "92520a205afefb021bd89a06a377a641a64a8072c9ae6a5c36f89fde9d637646",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/92520a205afefb021bd89a06a377a641a64a8072c9ae6a5c36f89fde9d637646.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/s/systemd/libsystemd0_245.4-4ubuntu3_arm64.deb"],
        deps = ["@ubuntu2004_libgcrypt20_aarch64//:all_files", "@ubuntu2004_liblz4-1_aarch64//:all_files", "@ubuntu2004_liblzma5_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libtasn1-6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2976e71392eddb28075d0711fa6099aa8b69fe4fe4c56512152cdfed601b3e8f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2976e71392eddb28075d0711fa6099aa8b69fe4fe4c56512152cdfed601b3e8f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libt/libtasn1-6/libtasn1-6_4.16.0-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libthai-data_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4789eb1d23fc72f9b1cb18649f5dede84442bce2ceb829b0f371aa248ba51405",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4789eb1d23fc72f9b1cb18649f5dede84442bce2ceb829b0f371aa248ba51405.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libt/libthai/libthai-data_0.1.28-3_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libthai0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "44844a07dca38de9b115055627aae8102d1eee741ef2cd6d55c181e7f45e4df2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/44844a07dca38de9b115055627aae8102d1eee741ef2cd6d55c181e7f45e4df2.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libt/libthai/libthai0_0.1.28-3_arm64.deb"],
        deps = ["@ubuntu2004_libthai-data_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libtiff5_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ebef8f1fdef883d26abd076b801b4a130ecf387b8837d9e0d549e801818c7a94",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ebef8f1fdef883d26abd076b801b4a130ecf387b8837d9e0d549e801818c7a94.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/t/tiff/libtiff5_4.1.0+git191117-2build1_arm64.deb"],
        deps = ["@ubuntu2004_libjbig0_aarch64//:all_files", "@ubuntu2004_liblzma5_aarch64//:all_files", "@ubuntu2004_libwebp6_aarch64//:all_files", "@ubuntu2004_libzstd1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libtinfo6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "65ae83568c287ec310ffd601cc34c07df5e86c91f4ea55742ba16db029b607fa",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/65ae83568c287ec310ffd601cc34c07df5e86c91f4ea55742ba16db029b607fa.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/n/ncurses/libtinfo6_6.2-0ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_libc6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libtsan0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8628f772f766521e343159a879616d0f6de6bbf067800adbdaf73a870bd41263",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8628f772f766521e343159a879616d0f6de6bbf067800adbdaf73a870bd41263.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/libtsan0_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libubsan1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ce36247c54345fca196ac6b32905e9721e9291d6577f5f79828d81ddd35eec48",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ce36247c54345fca196ac6b32905e9721e9291d6577f5f79828d81ddd35eec48.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/g/gcc-10/libubsan1_10-20200411-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_gcc-10-base_aarch64//:all_files", "@ubuntu2004_libgcc-s1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libudev1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e298137639d6ae27f4ca3a7ecf82e6de4a39c2d43b379b33b4ca7a46eb6336d4",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e298137639d6ae27f4ca3a7ecf82e6de4a39c2d43b379b33b4ca7a46eb6336d4.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/s/systemd/libudev1_245.4-4ubuntu3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libunistring2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "22323af293a1d3377ba77b717c683a267c37d3da095798edf098aaa23e5a5439",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/22323af293a1d3377ba77b717c683a267c37d3da095798edf098aaa23e5a5439.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libu/libunistring/libunistring2_0.9.10-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libunwind-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "34cdf9a7752343738e61f4788698a0a823f15d74072cd8d1bc982ce9b4e1ca16",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/34cdf9a7752343738e61f4788698a0a823f15d74072cd8d1bc982ce9b4e1ca16.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libu/libunwind/libunwind-dev_1.2.1-9build1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libunwind8_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "55dfa546c520734006575c3bd0e0992926311edb078856316825debcc1a45395",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/55dfa546c520734006575c3bd0e0992926311edb078856316825debcc1a45395.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libu/libunwind/libunwind8_1.2.1-9build1_arm64.deb"],
        deps = ["@ubuntu2004_liblzma5_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libuuid1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "03f3e3b532fd627072c97a476c3c9c0f1f1c52338741b85c0eaca6e9e0d66507",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/03f3e3b532fd627072c97a476c3c9c0f1f1c52338741b85c0eaca6e9e0d66507.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/u/util-linux/libuuid1_2.34-0.1ubuntu9_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libva-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "15e56e74b30c3650cb464c4be0fdd491556b5448df7bf1131d475163a8f03c48",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/15e56e74b30c3650cb464c4be0fdd491556b5448df7bf1131d475163a8f03c48.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/universe/libv/libva/libva-dev_2.7.0-2_arm64.deb"],
        deps = ["@ubuntu2004_libset-scalar-perl_aarch64//:all_files", "@ubuntu2004_libva-drm2_aarch64//:all_files", "@ubuntu2004_libva-glx2_aarch64//:all_files", "@ubuntu2004_libva-wayland2_aarch64//:all_files", "@ubuntu2004_libva-x11-2_aarch64//:all_files", "@ubuntu2004_libva2_aarch64//:all_files", "@ubuntu2004_libwayland-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libva-drm2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "26bde3cc9783f2c38d5915095ae3ebf754ec33a9a9370b3f138e72e7caf91d76",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/26bde3cc9783f2c38d5915095ae3ebf754ec33a9a9370b3f138e72e7caf91d76.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/universe/libv/libva/libva-drm2_2.7.0-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files", "@ubuntu2004_libva2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libva-glx2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "eb5d02e799fedd8ffaf6f7a217ac70351181cbc5a217285ab65b809ddae44518",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/eb5d02e799fedd8ffaf6f7a217ac70351181cbc5a217285ab65b809ddae44518.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/universe/libv/libva/libva-glx2_2.7.0-2_arm64.deb"],
        deps = ["@ubuntu2004_libgl1_aarch64//:all_files", "@ubuntu2004_libva-x11-2_aarch64//:all_files", "@ubuntu2004_libva2_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libva-wayland2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "75a2af56b861ee5fa00e7a815c84ff65a09bd0cc41498718ebb85065fe995c27",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/75a2af56b861ee5fa00e7a815c84ff65a09bd0cc41498718ebb85065fe995c27.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/universe/libv/libva/libva-wayland2_2.7.0-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files", "@ubuntu2004_libva2_aarch64//:all_files", "@ubuntu2004_libwayland-client0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libva-x11-2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4f474d6b37a306dcdada7d1e06879497a97bb4712cab228dc9f52ce2cc0b1440",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4f474d6b37a306dcdada7d1e06879497a97bb4712cab228dc9f52ce2cc0b1440.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/universe/libv/libva/libva-x11-2_2.7.0-2_arm64.deb"],
        deps = ["@ubuntu2004_libdrm2_aarch64//:all_files", "@ubuntu2004_libva2_aarch64//:all_files", "@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files", "@ubuntu2004_libxfixes3_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libva2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "41e966691e9366db7b335f0ca76d7926176cc6847941f7048ed75cf35e2ef96b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/41e966691e9366db7b335f0ca76d7926176cc6847941f7048ed75cf35e2ef96b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/universe/libv/libva/libva2_2.7.0-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libwayland-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "26023ef475f50d30126f999cd429b86e3b3072e0b4ff25580e7195dfe762217e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/26023ef475f50d30126f999cd429b86e3b3072e0b4ff25580e7195dfe762217e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/w/wayland/libwayland-bin_1.18.0-1_arm64.deb"],
        deps = ["@ubuntu2004_libexpat1_aarch64//:all_files", "@ubuntu2004_libxml2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libwayland-client0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8e2a40a07abd466a06db56ea1dff8fe92f8344a31baca3ca0a30912707a65f38",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8e2a40a07abd466a06db56ea1dff8fe92f8344a31baca3ca0a30912707a65f38.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/w/wayland/libwayland-client0_1.18.0-1_arm64.deb"],
        deps = ["@ubuntu2004_libffi7_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libwayland-cursor0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "01a09497261244fba1440a4c60a9b116bda0bc7a020434dd17adf0f76afaf37e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/01a09497261244fba1440a4c60a9b116bda0bc7a020434dd17adf0f76afaf37e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/w/wayland/libwayland-cursor0_1.18.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libwayland-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a4b4ca33fecfdd57e6e63820bc44fcf847cea1920c868599fc7faa475d32495a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a4b4ca33fecfdd57e6e63820bc44fcf847cea1920c868599fc7faa475d32495a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/w/wayland/libwayland-dev_1.18.0-1_arm64.deb"],
        deps = ["@ubuntu2004_libwayland-bin_aarch64//:all_files", "@ubuntu2004_libwayland-client0_aarch64//:all_files", "@ubuntu2004_libwayland-cursor0_aarch64//:all_files", "@ubuntu2004_libwayland-egl1_aarch64//:all_files", "@ubuntu2004_libwayland-server0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libwayland-egl1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f9dca70663b7e14adbfc4d78aaacc70ec65704fa61db570b61179f2b76ce86f7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f9dca70663b7e14adbfc4d78aaacc70ec65704fa61db570b61179f2b76ce86f7.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/w/wayland/libwayland-egl1_1.18.0-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libwayland-server0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "01a55420cd1bc03b3129ea484aebec65a239293d970026ccc05b6fc1121de85e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/01a55420cd1bc03b3129ea484aebec65a239293d970026ccc05b6fc1121de85e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/w/wayland/libwayland-server0_1.18.0-1_arm64.deb"],
        deps = ["@ubuntu2004_libffi7_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libwebp6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "45a09de494dfc849f37bb06efb25aab841bcd5b7d13829268e21412c64a7d18f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/45a09de494dfc849f37bb06efb25aab841bcd5b7d13829268e21412c64a7d18f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libw/libwebp/libwebp6_0.6.1-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libx11-6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "f94c9798948279eaa4c3a3112347e664631acc9f537dff70c8cd3cf97659ab7f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/f94c9798948279eaa4c3a3112347e664631acc9f537dff70c8cd3cf97659ab7f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libx11/libx11-6_1.6.9-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libx11-data_aarch64//:all_files", "@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libx11-data_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ddcf4113c2157f47198a62a2bf309b458538f36d2bf21222e54a5ee03fc1035c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ddcf4113c2157f47198a62a2bf309b458538f36d2bf21222e54a5ee03fc1035c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libx11/libx11-data_1.6.9-2ubuntu1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libx11-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b79885f08812a13979d057e2612232c750ba451a92a7373c577863607762daa8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b79885f08812a13979d057e2612232c750ba451a92a7373c577863607762daa8.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libx11/libx11-dev_1.6.9-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxau-dev_aarch64//:all_files", "@ubuntu2004_libxcb1-dev_aarch64//:all_files", "@ubuntu2004_libxdmcp-dev_aarch64//:all_files", "@ubuntu2004_x11proto-dev_aarch64//:all_files", "@ubuntu2004_xtrans-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libx11-xcb1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4d727b39abf7ac08d8db5db829339976c9e74a86b38dc68501d29dee1d1eb4b1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4d727b39abf7ac08d8db5db829339976c9e74a86b38dc68501d29dee1d1eb4b1.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libx11/libx11-xcb1_1.6.9-2ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxau-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "69c8db9a78e55302ba359b889c1be9d0c901d7bc13cef8776cb9119f5125b465",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/69c8db9a78e55302ba359b889c1be9d0c901d7bc13cef8776cb9119f5125b465.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxau/libxau-dev_1.0.9-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libxau6_aarch64//:all_files", "@ubuntu2004_x11proto-core-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxau6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0f19eb67992cf9a755a781502529750841d93fb8729691df02b9b16aec845b40",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0f19eb67992cf9a755a781502529750841d93fb8729691df02b9b16aec845b40.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxau/libxau6_1.0.9-0ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb-dri2-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e7d1196d103cf5a7429b8a703ead8fddd5e0460a4728bf0a05a2b7fc2130ef5c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e7d1196d103cf5a7429b8a703ead8fddd5e0460a4728bf0a05a2b7fc2130ef5c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb-dri2-0_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb-dri3-0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "bac02c27484164dec151943a1b7c4cc0d0ae05545c557b570393a9940e075ef0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bac02c27484164dec151943a1b7c4cc0d0ae05545c557b570393a9940e075ef0.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb-dri3-0_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb-glx0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "b638891eee30451c8148e2c6eb3523bdf629542f0d9bf9c748d6a1e3c45106a9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/b638891eee30451c8148e2c6eb3523bdf629542f0d9bf9c748d6a1e3c45106a9.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb-glx0_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb-present0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "bad56c61123773606cb44c9cacbacab4ec5a71350084278de5f2a4da4a01f7fb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/bad56c61123773606cb44c9cacbacab4ec5a71350084278de5f2a4da4a01f7fb.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb-present0_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb-render0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4a82f349fc869c8f803c73fdd7e94a6ee1c06d6fb3f6ceab1227efee64828957",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4a82f349fc869c8f803c73fdd7e94a6ee1c06d6fb3f6ceab1227efee64828957.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb-render0_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb-shm0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e0eb8cad040933946459ff6d2116953a68ce149e9d695c28552a1d0beaa9a07b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e0eb8cad040933946459ff6d2116953a68ce149e9d695c28552a1d0beaa9a07b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb-shm0_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb-sync1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "835b88e97486db5e303a89f9d07a530ac182cb5b0b102776cf76b56a00340504",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/835b88e97486db5e303a89f9d07a530ac182cb5b0b102776cf76b56a00340504.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb-sync1_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb-xfixes0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d334081b50fea5a3ca31e35efc8915b15d183d448fd9f8aa15dee05ff1c9fc1b",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d334081b50fea5a3ca31e35efc8915b15d183d448fd9f8aa15dee05ff1c9fc1b.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb-xfixes0_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb1-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3d69db8f1f044a48ab5fe0c3e32c3996d1bc5d2f8cae61022d17c3d55bf9b011",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3d69db8f1f044a48ab5fe0c3e32c3996d1bc5d2f8cae61022d17c3d55bf9b011.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb1-dev_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libpthread-stubs0-dev_aarch64//:all_files", "@ubuntu2004_libxcb1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcb1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "88e171a33249719366a4e29330775e2ac777226f1f379557daf0777083a1fc93",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/88e171a33249719366a4e29330775e2ac777226f1f379557daf0777083a1fc93.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcb/libxcb1_1.14-2_arm64.deb"],
        deps = ["@ubuntu2004_libxau6_aarch64//:all_files", "@ubuntu2004_libxdmcp6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcomposite1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "61407cb59f0fe5e0f04ea3718a14f9da13a97c0b4675746289e9bd45ec9cc5b8",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/61407cb59f0fe5e0f04ea3718a14f9da13a97c0b4675746289e9bd45ec9cc5b8.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcomposite/libxcomposite1_0.4.5-1_arm64.deb"],
        deps = ["@ubuntu2004_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxcursor1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "16dbb79ba4a78e78e85edba61cbad2503040cc879b297ea8e5b047f8bcda3b9d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/16dbb79ba4a78e78e85edba61cbad2503040cc879b297ea8e5b047f8bcda3b9d.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxcursor/libxcursor1_1.2.0-2_arm64.deb"],
        deps = ["@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxfixes3_aarch64//:all_files", "@ubuntu2004_libxrender1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxdamage1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "fabd968fe71f4c5ce1af2aa2ee0ef98e15fe64c17d206a6ceba29d0dc52f124e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fabd968fe71f4c5ce1af2aa2ee0ef98e15fe64c17d206a6ceba29d0dc52f124e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxdamage/libxdamage1_1.1.5-2_arm64.deb"],
        deps = ["@ubuntu2004_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxdmcp-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a9eac5abd4448b231934e8bf6a73336c708ffeabe71b18092c2629af870d4179",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a9eac5abd4448b231934e8bf6a73336c708ffeabe71b18092c2629af870d4179.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxdmcp/libxdmcp-dev_1.1.3-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libxdmcp6_aarch64//:all_files", "@ubuntu2004_x11proto-core-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxdmcp6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "065c2e430e0de48007107aabb9b6ee25bb360eecca312c426827e158fa10974d",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/065c2e430e0de48007107aabb9b6ee25bb360eecca312c426827e158fa10974d.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxdmcp/libxdmcp6_1.1.3-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libbsd0_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxext6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4f153b62b8c5f77a8b043d7b6746ed43deedf0c2dee5d3911600df89b1a7c3db",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4f153b62b8c5f77a8b043d7b6746ed43deedf0c2dee5d3911600df89b1a7c3db.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxext/libxext6_1.3.4-0ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxfixes3_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3cea7f92d5ff7b0b2a29264c74181edf525b278b0be8d82a5ffe73cf20b1aa1e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3cea7f92d5ff7b0b2a29264c74181edf525b278b0be8d82a5ffe73cf20b1aa1e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxfixes/libxfixes3_5.0.3-2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxi6_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a0c99bb61d3881f7d516fc154201b1bc0de056201b01672301c14515242557dc",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a0c99bb61d3881f7d516fc154201b1bc0de056201b01672301c14515242557dc.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxi/libxi6_1.7.10-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxinerama1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a2e16ba9ad2697a35006e377e1e45c192776513ac3529b189a82614938c6a811",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a2e16ba9ad2697a35006e377e1e45c192776513ac3529b189a82614938c6a811.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxinerama/libxinerama1_1.1.4-2_arm64.deb"],
        deps = ["@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxkbcommon0_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "3f2620fa08f83a1d139dde4012c49519a96abd58f07a92945e0f16154096bb2a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/3f2620fa08f83a1d139dde4012c49519a96abd58f07a92945e0f16154096bb2a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxkbcommon/libxkbcommon0_0.10.0-1_arm64.deb"],
        deps = ["@ubuntu2004_xkb-data_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxml2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "6242892cb032859044ddfcfbe61bac5678a95c585d8fff4525acaf45512e3d39",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/6242892cb032859044ddfcfbe61bac5678a95c585d8fff4525acaf45512e3d39.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxml2/libxml2_2.9.10+dfsg-5_arm64.deb"],
        deps = ["@ubuntu2004_libicu66_aarch64//:all_files", "@ubuntu2004_liblzma5_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxrandr2_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "32d7f174ee845be1c7c4198113b401520f5bd77e4cf09b534c53eca0e496dc21",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/32d7f174ee845be1c7c4198113b401520f5bd77e4cf09b534c53eca0e496dc21.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxrandr/libxrandr2_1.5.2-0ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files", "@ubuntu2004_libxrender1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxrender1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "931f3d61ec6528221a0a2dc6f9fdac494e24d7e92e49ce1811c495571f3b2cea",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/931f3d61ec6528221a0a2dc6f9fdac494e24d7e92e49ce1811c495571f3b2cea.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxrender/libxrender1_0.9.10-1_arm64.deb"],
        deps = ["@ubuntu2004_libx11-6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxshmfence1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "41a76da1aa2ffd152820e8996f7366d5525003e8633d1721b8ee326321d345e9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/41a76da1aa2ffd152820e8996f7366d5525003e8633d1721b8ee326321d345e9.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxshmfence/libxshmfence1_1.3-1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libxxf86vm1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "074d74280211f2498ca727aa04902ee91aa53eee550db5901a930ec5b54c0b82",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/074d74280211f2498ca727aa04902ee91aa53eee550db5901a930ec5b54c0b82.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libx/libxxf86vm/libxxf86vm1_1.1.4-1build1_arm64.deb"],
        deps = ["@ubuntu2004_libx11-6_aarch64//:all_files", "@ubuntu2004_libxext6_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_libzstd1_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "ca6c31a967c02ed6f171e21ddbfbe1baba174dc2c550d32b047faaad7c7bd3b0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/ca6c31a967c02ed6f171e21ddbfbe1baba174dc2c550d32b047faaad7c7bd3b0.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/libz/libzstd/libzstd1_1.4.4+dfsg-3_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_linux-libc-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "960d3d3d98ca3363878ac99d602c40ddb5e5f7e586a5f63cdfddcfe77595807f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/960d3d3d98ca3363878ac99d602c40ddb5e5f7e586a5f63cdfddcfe77595807f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/l/linux/linux-libc-dev_5.4.0-26.30_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_make_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4856ecbb79f91dd33f46474fca70331cc0940f7ba79a182e4e93970473c5a3b0",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4856ecbb79f91dd33f46474fca70331cc0940f7ba79a182e4e93970473c5a3b0.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/make-dfsg/make_4.2.1-1.2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_mesa-common-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "65aeee78437580b4125cff39f7a87d1ec40e8f638ba589f5886be3b234582eb5",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/65aeee78437580b4125cff39f7a87d1ec40e8f638ba589f5886be3b234582eb5.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/m/mesa/mesa-common-dev_20.0.4-2ubuntu1_arm64.deb"],
        deps = ["@ubuntu2004_libdrm-dev_aarch64//:all_files", "@ubuntu2004_libgl-dev_aarch64//:all_files", "@ubuntu2004_libglx-dev_aarch64//:all_files", "@ubuntu2004_libx11-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_ncurses-bin_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9fd4628481fda4001d580800a4eb6a23cee59aaa6f3343d9ae5d9fa0d67621e2",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9fd4628481fda4001d580800a4eb6a23cee59aaa6f3343d9ae5d9fa0d67621e2.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/n/ncurses/ncurses-bin_6.2-0ubuntu2_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_openssl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "c4921b99c49b66548c0b5670b22bffcecfbfe067b46a95e0c707e9d25b9b6dae",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/c4921b99c49b66548c0b5670b22bffcecfbfe067b46a95e0c707e9d25b9b6dae.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/o/openssl/openssl_1.1.1f-1ubuntu2_arm64.deb"],
        deps = ["@ubuntu2004_libc6_aarch64//:all_files", "@ubuntu2004_libssl1.1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_passwd_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "05ed9ae975b864b4d3912b37b08814f1c2e971fd10079ad78669e5be2c103056",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/05ed9ae975b864b4d3912b37b08814f1c2e971fd10079ad78669e5be2c103056.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/s/shadow/passwd_4.8.1-1ubuntu5_arm64.deb"],
        deps = ["@ubuntu2004_libaudit1_aarch64//:all_files", "@ubuntu2004_libcrypt1_aarch64//:all_files", "@ubuntu2004_libpam-modules_aarch64//:all_files", "@ubuntu2004_libpam0g_aarch64//:all_files", "@ubuntu2004_libselinux1_aarch64//:all_files", "@ubuntu2004_libsemanage1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_patch_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "d80701e62a418ab4229cbed2c23603cd5d49268ca24583377a35ea1e6c91808a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/d80701e62a418ab4229cbed2c23603cd5d49268ca24583377a35ea1e6c91808a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/patch/patch_2.7.6-6_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_perl-base_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e7806567c36e07a8b84cb2ea762a29feaa88346a2c0b7d719e5515299cff687c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e7806567c36e07a8b84cb2ea762a29feaa88346a2c0b7d719e5515299cff687c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/perl/perl-base_5.30.0-9build1_arm64.deb"],
        deps = ["@ubuntu2004_libcrypt1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_perl-modules-5.30_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "85bf7d2a1e1fba022a1f246d4c60439b987f5b858b42207759b5cac5b67f9896",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/85bf7d2a1e1fba022a1f246d4c60439b987f5b858b42207759b5cac5b67f9896.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/perl/perl-modules-5.30_5.30.0-9build1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_perl_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "141ce1673064c8d2f7ea55529bc0571e91e9c913ecca55934ecee894d2677e5e",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/141ce1673064c8d2f7ea55529bc0571e91e9c913ecca55934ecee894d2677e5e.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/p/perl/perl_5.30.0-9build1_arm64.deb"],
        deps = ["@ubuntu2004_libperl5.30_aarch64//:all_files", "@ubuntu2004_perl-base_aarch64//:all_files", "@ubuntu2004_perl-modules-5.30_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_readline-common_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "38c3ac67e2dab4122a2f948f433c4cb5d5653d82b323f3ff30599797b7adee9f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/38c3ac67e2dab4122a2f948f433c4cb5d5653d82b323f3ff30599797b7adee9f.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/r/readline/readline-common_8.0-4_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_sed_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "e52211cf612f243f32f1f31b7dd05ea9bad98a89854736003bec6461711667ae",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/e52211cf612f243f32f1f31b7dd05ea9bad98a89854736003bec6461711667ae.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/s/sed/sed_4.7-1_arm64.deb"],
        deps = ["@ubuntu2004_libacl1_aarch64//:all_files", "@ubuntu2004_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_sensible-utils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "9ef20f3a2c2eac2d6d80b9ee0550f315d21fc7bc9e643f2f1b1e94c93f444601",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/9ef20f3a2c2eac2d6d80b9ee0550f315d21fc7bc9e643f2f1b1e94c93f444601.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/s/sensible-utils/sensible-utils_0.0.12+nmu1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_shared-mime-info_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "8982aee508496b7a13dbf6b14c6fe1e7c0fcfc7d22fe2da1ad9cf40e248e142c",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/8982aee508496b7a13dbf6b14c6fe1e7c0fcfc7d22fe2da1ad9cf40e248e142c.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/s/shared-mime-info/shared-mime-info_1.15-1_arm64.deb"],
        deps = ["@ubuntu2004_libglib2.0-0_aarch64//:all_files", "@ubuntu2004_libxml2_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_tar_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "fc5b5446e019d2fc397d614b1221bb04d87f2390d54531d1802f8444abea3da1",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/fc5b5446e019d2fc397d614b1221bb04d87f2390d54531d1802f8444abea3da1.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/t/tar/tar_1.30+dfsg-7_arm64.deb"],
        deps = ["@ubuntu2004_libacl1_aarch64//:all_files", "@ubuntu2004_libselinux1_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_tzdata_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "aa2f203d74bc18d947ca8ccdaa231d77407763479eb3df18cf1eec7299e65819",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/aa2f203d74bc18d947ca8ccdaa231d77407763479eb3df18cf1eec7299e65819.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/t/tzdata/tzdata_2019c-3ubuntu1_all.deb"],
        deps = ["@ubuntu2004_debconf_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_ubuntu-mono_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "377778173059a13ea1a41c93d6186675da5750568ddd6e6f0bd6c758a9713ac7",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/377778173059a13ea1a41c93d6186675da5750568ddd6e6f0bd6c758a9713ac7.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/u/ubuntu-themes/ubuntu-mono_19.04-0ubuntu3_all.deb"],
        deps = ["@ubuntu2004_hicolor-icon-theme_aarch64//:all_files", "@ubuntu2004_humanity-icon-theme_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_ucf_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "0fd6782ac99f3d9814960edefd5cceaa84f3a33feec6a149f7edf59431f3a3ac",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/0fd6782ac99f3d9814960edefd5cceaa84f3a33feec6a149f7edf59431f3a3ac.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/u/ucf/ucf_3.0038+nmu1_all.deb"],
        deps = ["@ubuntu2004_coreutils_aarch64//:all_files", "@ubuntu2004_debconf_aarch64//:all_files", "@ubuntu2004_sensible-utils_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_x11proto-core-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "a952dce35c7b06b3a0e32d7bb16eb2c3922bdaa977813200b09223daabbee42a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/a952dce35c7b06b3a0e32d7bb16eb2c3922bdaa977813200b09223daabbee42a.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/x/xorgproto/x11proto-core-dev_2019.2-1ubuntu1_all.deb"],
        deps = ["@ubuntu2004_x11proto-dev_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_x11proto-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "4144072931cbfbb422b465ae4775ce906d01ea816d432ed820b301e08cfef975",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/4144072931cbfbb422b465ae4775ce906d01ea816d432ed820b301e08cfef975.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/x/xorgproto/x11proto-dev_2019.2-1ubuntu1_all.deb"],
        deps = ["@ubuntu2004_xorg-sgml-doctools_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_xkb-data_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "890cc198b5a1023e3b3a879448524267d338b046a415bae7aff6216165bc4085",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/890cc198b5a1023e3b3a879448524267d338b046a415bae7aff6216165bc4085.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/x/xkeyboard-config/xkb-data_2.29-2_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_xorg-sgml-doctools_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2f6463489813c2a08e077a6502453c3252453f7cbdab9f323006e081b33e7ad3",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2f6463489813c2a08e077a6502453c3252453f7cbdab9f323006e081b33e7ad3.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/x/xorg-sgml-doctools/xorg-sgml-doctools_1.11-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_xtrans-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "45277c51d5d83db351b61859314b59595c9626ac372fbb2fc0d5542e169d9086",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/45277c51d5d83db351b61859314b59595c9626ac372fbb2fc0d5542e169d9086.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/x/xtrans/xtrans-dev_1.4.0-1_all.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_xz-utils_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "619d2a859a48882be05a056ca99967e1adabc0134a5fb7385f0cb1b29e229a92",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/619d2a859a48882be05a056ca99967e1adabc0134a5fb7385f0cb1b29e229a92.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/x/xz-utils/xz-utils_5.2.4-1_arm64.deb"],
        deps = ["@ubuntu2004_liblzma5_aarch64//:all_files"],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_zlib1g-dev_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "2ac4bc3325af19004eb378aeb18298df51b6f51f2bacd230fcd8a7f4d41a68a6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/2ac4bc3325af19004eb378aeb18298df51b6f51f2bacd230fcd8a7f4d41a68a6.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/z/zlib/zlib1g-dev_1.2.11.dfsg-2ubuntu1_arm64.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "ubuntu2004_zlib1g_aarch64",
        exclude_paths = ["usr/share/man", "usr/share/doc", "usr/share/icons", "usr/share/ca-certificates/mozilla/NetLock_Arany_=Class_Gold=_F\305\221tan\303\272s\303\255tv\303\241ny.crt", "usr/bin/X11", "lib/systemd/system/system-systemd\\x2dcryptsetup.slice"],
        sha256 = "eb740529da605507e7ae9e9263641bb49dd03de0568446f09b32e6341ca24778",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/deb-mirrors/eb740529da605507e7ae9e9263641bb49dd03de0568446f09b32e6341ca24778.deb", "https://snapshot.ubuntu.com/ubuntu/20230401T000000Z/pool/main/z/zlib/zlib1g_1.2.11.dfsg-2ubuntu1_arm64.deb"],
        deps = [],
    )
